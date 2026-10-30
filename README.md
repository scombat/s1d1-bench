# s1d1-bench

Banc ouvert pour mesurer à quel point un classifieur de prompts s'accorde avec des juges LLM locaux, sur la grille de
notation v7 (palier de complexité, score, tâche, `needs_tools`).

## Ce que le banc mesure, et ce qu'il ne mesure pas

- **Mesuré.** L'accord des quatre sorties d'un classifieur avec la médiane des juges : palier de complexité, score,
  tâche et `needs_tools`, globalement puis par langue, par source et par palier. Les témoins aléatoire et longueur du
  prompt donnent le plancher, l'accord de chaque juge avec les autres donne le plafond.
- **Non mesuré : la grille.** Si la grille elle-même correspond à « quel modèle suffit » relève d'un autre banc, celui
  des réponses, qui n'est pas traité ici.
- **Non mesuré : le routage.** Le mapping score vers modèle est l'affaire du déployeur.
- **Non mesuré : la distribution réelle.** Le jeu est construit par cases, `expert` sera la moins fournie ; les prompts
  agentiques et multi-tours sont absents, le jeu est fait de premiers prompts.
- **Non mesuré : les autres langues.** Seules les quatre langues choisies sont couvertes.

## Les commandes

- `bench pool` : construit le jeu de prompts à partir de jeux publics du Hub.
- `bench export` : exporte les prompts à faire juger.
- `bench judge` : fait juger les prompts par un modèle local (MLX) avec la grille v7.
- `bench validate` : vérifie qu'un dossier de juge est complet et cohérent avec les prompts.
- `bench agree` : calcule l'accord deux à deux entre des dossiers de juges (tableau en markdown).
- `bench draw` : tire les prompts de base par strates (tirage aléatoire à graine fixe sur un dossier de pré-tri).
- `bench translate` : traduit un fichier de prompts dans une langue (fra, rus, zho) avec un modèle local (MLX).
- `bench witness` : récupère les traductions m-ArenaHard (Google) des prompts Arena-Hard d'un jeu, comme témoins.
- `bench check` : signale les traductions à relire (`structure`, `labels`, `request`) et écrit la feuille de relecture (`review`).

Installation : `uv sync --group pool --group dev` (sur Apple silicon, pour faire tourner un juge :
`uv sync --group pool --group judge --group dev`), puis `uv run bench --help`.

## Reproduire l'essai

L'essai fait juger 50 prompts par trois modèles locaux, chacun avec le raisonnement coupé (`off`) et plafonné à 512
jetons (`cap:512`), soit six exécutions ; cinq dossiers `judges/trial-*` sont gardés (gpt-oss `off` n'a pas pu étiqueter un prompt et a été retiré). Les résultats et les durées sont dans
[judges/TRIAL.md](judges/TRIAL.md) : c'est un essai de l'outillage sur 50 prompts, pas un résultat du banc. Les trois
modèles, leurs révisions et la façon dont `bench judge` coupe ou plafonne leur raisonnement sont dans
[judges/MODELS.md](judges/MODELS.md) ; le format des dossiers de juges est dans [judges/README.md](judges/README.md).

Toutes les commandes se lancent depuis la racine du dépôt. Les étapes 4 et 5 demandent un Mac Apple silicon et
`uv sync --group pool --group judge --group dev` (ou `uv run --group judge`), un seul modèle tourne à la fois.

1. **Liste d'exclusion** (`data/exclusions/s1d1-seen-keys.txt`). Elle est produite par le dépôt de recherche privé
   s1d1-model-router, sur sa branche `seen-keys` (`pipeline/13_seen_keys.py`). La commande se lance dans ce dépôt de
   recherche ; `--out` est le chemin de la liste dans ce dépôt-ci (écrit ici en relatif, à adapter) :

   ```
   uv run python -I pipeline/13_seen_keys.py --inputs 'dataset/final/*.parquet' \
       --out data/exclusions/s1d1-seen-keys.txt
   ```

   Sans accès à ce dépôt, le fichier déjà commité suffit : [data/exclusions/README.md](data/exclusions/README.md) dit
   ce qu'il contient (des clés de hachage, aucun texte de prompt).

2. **Le jeu de prompts.** Jeux publics du Hub, sans jeton (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` est posé par la commande
   elle-même) :

   ```
   uv run --group pool bench pool
   ```

   Sortie : `data/pool/pool.parquet`, `data/pool/pool.stats.json`, `data/pool/candidates-eng.jsonl`.

3. **Les 50 prompts de l'essai.** Le tirage (graine 0 ; 17 Dolly, 17 OASST2 puis 16 Arena-Hard)
   écrit la liste de clés `data/prompts/trial-50.select.txt`, puis `bench export` fabrique le fichier de prompts :

   ```
   uv run python -I - <<'PY'
   import json
   import random

   import pyarrow.parquet as pq

   source_of = dict(zip(*pq.read_table("data/pool/pool.parquet", columns=["prompt_key", "source"]).to_pydict().values()))
   by_source: dict[str, list[str]] = {}
   for line in open("data/pool/candidates-eng.jsonl", encoding="utf-8"):
       key = json.loads(line)["id"]
       by_source.setdefault(source_of[key], []).append(key)
   rng = random.Random(0)
   quota = {"dolly": 17, "oasst2": 17, "arena_hard": 16}
   keys = []
   for source, n in quota.items():
       candidates = sorted(by_source[source])
       keys += rng.sample(candidates, n)
   assert len(keys) == len(set(keys)) == 50
   open("data/prompts/trial-50.select.txt", "w", encoding="utf-8").write("\n".join(keys) + "\n")
   PY

   uv run bench export --pool data/pool/pool.parquet --select data/prompts/trial-50.select.txt --langs eng \
       --name trial-50 --seed 0
   ```

4. **Les modèles**, dans `~/mlx-models/<org>--<nom>` (jamais dans un dépôt). `bench judge` affiche la commande exacte
   s'il en manque un ; les identifiants et les révisions à utiliser sont dans [judges/MODELS.md](judges/MODELS.md) :

   ```
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --group pool hf download <org>/<nom> --revision <révision> \
       --local-dir ~/mlx-models/<org>--<nom>
   ```

   à faire pour `mlx-community/gemma-4-31b-it-8bit`, `mlx-community/Qwen3.8-27B-8bit` et
   `mlx-community/gpt-oss-20b-MXFP4-Q8`.

5. **Les juges**, l'un après l'autre (provider `local`, MLX, Apple silicon seulement). Pour chaque famille,
   `--thinking off` puis `--thinking cap:512` ; `--author` reçoit votre nom :

   ```
   export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
   uv run --group judge bench judge --model mlx-community/gpt-oss-20b-MXFP4-Q8 --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-gptoss-cap512 --author <nom> --thinking cap:512 --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/Qwen3.8-27B-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-qwen3-off --author <nom> --thinking off --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/Qwen3.8-27B-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-qwen3-cap512 --author <nom> --thinking cap:512 --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/gemma-4-31b-it-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-gemma4-off --author <nom> --thinking off --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/gemma-4-31b-it-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-gemma4-cap512 --author <nom> --thinking cap:512 --max-tokens 1024
   ```

   Une exécution interrompue se reprend avec `--resume`.

6. **Vérifier et comparer** :

   ```
   for d in judges/*/; do uv run bench validate "$d"; done
   uv run bench agree judges/trial-*
   ```

## Le tirage des prompts de base

Le jeu de base n'est pas choisi à la main (docs/32 § 3.1, amendé le 2026-10-29). Les 900 candidats anglais
(`data/prompts/presort-900`, export des `candidates-eng.jsonl`) sont étiquetés par un juge de pré-tri qui n'est pas
Gemma 4 (`judges/presort-qwen3-off`, Qwen3.8 27B sans réflexion ; c'est un dossier de juge ordinaire, pas un juge du
banc). `bench draw` tire ensuite 20 prompts par strate `trivial`, `low`, `medium`, `high` : mélange à graine fixe de
chaque strate, puis les `needs_tools` vrais en tourniquet jusqu'à 20, puis chaque tâche de la rubrique jusqu'à 4
(depuis la strate qui a le plus de place), puis remplissage. Un veto (`<id><TAB><raison>` par ligne, deux règles seulement :
même gabarit dans la même strate, ou traduction qui changerait la tâche) retire un id de son ordre, le suivant de la
strate prend sa place et les autres tirages ne bougent pas. Ce que les candidats ne permettent pas (un minimum de
`needs_tools` ou de tâche) est signalé, pas corrigé en silence.

```
uv run bench draw --presort judges/presort-qwen3-off --meta data/prompts/presort-900.meta.parquet \
    --out data/prompts/base-v1.select.txt --seed 0 --veto data/prompts/base-v1.veto.txt
```

`base-v1.select.txt` (les clés tirées), `base-v1.draw.json` (règle, graine, strates, manques) et `base-v1.veto.txt`
(les retraits et leur raison) sont publiés : le tirage se rejoue. La case `expert` est faite des dix prompts de
`data/handwritten/expert-v1.jsonl`, pas du vivier.

## La traduction

Les 90 prompts de base anglais (`data/prompts/base-v1-eng`, export de `base-v1.select.txt` et des dix prompts écrits à la
main) sont traduits en français, russe et chinois simplifié par le modèle du pré-tri, Qwen3.8 27B 8 bits sans réflexion,
à température 0 (docs/33). `bench translate` traduit un fichier de prompts dans une langue, un prompt par génération :
la consigne (`consigne/translate-v1-<lang>.md`) est le message système, le prompt est donné entre balises `<prompt>`
comme une donnée.

```
uv run --group judge bench translate --model mlx-community/Qwen3.8-27B-8bit --prompts data/prompts/base-v1-eng.jsonl \
    --lang fra --consigne consigne/translate-v1-fra.md --out data/translations/base-v1 --author <nom> --thinking off
```

Sortie dans `--out` : `<lang>.jsonl` (les traductions), `<lang>.meta.json` (modèle, révision, consigne et son SHA-256,
SHA-256 du fichier anglais, réglages), `<lang>.log.jsonl` (un état par prompt : `ok`, `empty`, `envelope`, `truncated`,
`error`). Comme pour un juge, les fichiers ne sont jamais réécrits ; `--resume` reprend une série interrompue avec les
mêmes réglages. Une traduction qui contient une ligne `</prompt>` ou qui commence par `<prompt` est une erreur consignée,
jamais une traduction ; seule exception, une réponse entièrement enveloppée dans une paire `<prompt>…</prompt>` (le
modèle recopie les balises) est dépouillée de cette paire et le journal le dit (`unwrapped`).

Vitesse mesurée (M-série 128 Go, MLX, série de 90 prompts, chargement compris) : 825 s en français, 825 s en russe,
657 s en chinois, soit une médiane de 9 à 11 jetons par seconde en décodage.

### Les vérifications

Une traduction n'est pas livrée sans relecture. `bench check` ne corrige rien : il signale des lignes pour qu'un humain
les lise, et écrit tout dans `data/checks/<jeu>/` (`<lang>.structure.jsonl`, `<lang>.labels.jsonl`,
`<lang>.request.jsonl`, puis `<lang>.review.md`, la feuille où chaque id signalé a un `avis :` et une `décision :`).

- `structure` : blocs de code, URL, nombres, guillemets droits, nombre de lignes, longueur (rapport traduction/anglais
  borné par langue), balise `<prompt>`. Règle de lecture : les signalements `quotes` sont presque tous des guillemets
  typographiques (« », 「」) à la place de guillemets droits, une limite de la règle et non une erreur ; les `numbers`
  sont des « 3:30 » devenus « 15h30 », des ordinaux écrits en toutes lettres ou des séparateurs de milliers ; les
  `length` sont de très courts prompts.
- `labels` : le juge Qwen3 (celui du pré-tri) étiquette l'anglais et la traduction (`judges/check-qwen3-off-<lang>`) ;
  une `task` ou un `needs_tools` qui change est signalé. Pour les 28 prompts Arena-Hard, la traduction Google
  (m-ArenaHard) sert de témoin (`bench witness`, `judges/witness-qwen3-off-<lang>`) : si le témoin reçoit du juge la
  même étiquette que notre traduction, le signalement est levé. Le bruit du juge domine ce contrôle : presque toutes
  les différences d'étiquette sont du bruit, sauf trois mistraductions que Gemma 4 a aussi repérées.
- `request` : Gemma 4 31B (un autre modèle que le traducteur) lit l'anglais et la traduction et répond en JSON
  `{"same_request": bool, "reason": str}` (`consigne/check-same-request-v1.md`). Un `false` est un signal, pas un verdict.

```
uv run --group judge bench check request --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1 --model mlx-community/gemma-4-31b-it-8bit --author <nom> --thinking off
uv run bench check structure --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1
uv run bench check labels --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1 --labels-english judges/check-qwen3-off-eng \
    --labels judges/check-qwen3-off-fra --labels-witness judges/witness-qwen3-off-fra
uv run bench check review --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1 --witness data/translations/base-v1/witness-arena
```

Le témoin Google est brut : entités HTML (`&#39;`) et espaces en bord de ligne y restent, nous ne le nettoyons pas.

Une correction humaine est une superposition : `data/translations/base-v1/<lang>.corrections.jsonl` (une ligne
`{"id", "prompt", "reason", "author"}` par id corrigé). `bench export` l'applique et marque `corrected` dans
`meta.parquet`; les fichiers du modèle (`<lang>.jsonl`) ne sont jamais réécrits. Un veto (traduction qui changerait la
tâche, deux signaux réunis) retirerait l'id du tirage et ses remplaçants iraient dans une vague `-r1` (`--translations`
se répète, une fois par vague) ; la relecture de `base-v1` n'en a décidé aucun.

La relecture de `base-v1` (les trois langues lues contre l'anglais) a retenu 22 corrections : français 9, russe 6,
chinois 7. 11 changent le sens (les fautes de frappe anglaises « captiols » et « store » prises au pied de la lettre
dans les trois langues ; « banks' » perdu en français et en chinois ; « stochastic » ajouté en russe ; « aloud » ajouté
et « keep score » mal rendu en français), 7 sont des termes techniques faux (key schedule, responsable du traitement,
assignment en français ; shellcode en russe ; litmus test, proto-indo-européen et titre de revue traduit en chinois),
4 sont des noms ou des accords abîmés (Tanooki, un accord de grammaire, « Шлой », Bourbon rendu par Bourgogne). Aucun
veto. Ce que cela ne prouve pas : le contrôle de sens est un contrôle de lecture sur 90 prompts par langue, fait par un
seul relecteur (Claude Fable 5.1), pas par des locuteurs natifs ; ni la qualité stylistique ni le registre ne sont mesurés.

### Le jeu apparié

`data/prompts/base-v1.jsonl` réunit les 90 prompts dans les quatre langues (360 lignes, 9 lots, ids `<clé>-<lang>`),
avec les corrections appliquées :

```
uv run bench export --select data/prompts/base-v1.select.txt --handwritten data/handwritten/expert-v1.jsonl \
    --translations data/translations/base-v1 --langs eng,fra,rus,zho --name base-v1
```

Les fichiers `base-v1-<lang>.jsonl` sont ceux que les juges de vérification ont vus, avant corrections ; ils restent tels
quels (leur SHA-256 est celui que `bench validate` demande aux dossiers `check-*`), et `bench export` refuse de réécrire
un fichier de prompts qu'un dossier de juge épingle par son SHA-256 si le contenu change. `meta.parquet` porte `translator`
(`modèle@révision`, vide en anglais) et `corrected`.

## Les vérités

`bench run` réunit les dossiers de juges et de classifieurs d'un jeu en une table, une ligne par id ; les vérités ne
viennent que des juges principaux (`--main`). Le palier vrai est la médiane ordinale (trivial < low < medium < high <
expert) des votes non nuls, `truth_all` sur les 12 votes des quatre versions d'un prompt de base (3 juges × 4 langues),
`truth_lang` sur les 3 votes de l'id. Règle R5 (docs/34 § 0.3) : un nombre impair de votes prend celui du milieu ; un
nombre pair prend la moyenne des rangs des deux votes du milieu, arrondie vers le bas. C'est la médiane basse quand les
deux votes du milieu sont égaux ou voisins (6/6 sur deux paliers voisins donne le plus bas), mais low et high donnent
medium. Le palier est nul (`undecidable`) quand les `null` sont strictement majoritaires (7 sur 12 ; 6 sur 12 ne
l'est pas). `task` et `needs_tools` vont au vote majoritaire, l'égalité tranchée par le premier juge de `--main`.

Les abstentions : un juge dont la sortie ne donne pas d'étiquette lisible (refus, JSON illisible, réponse coupée :
`json_error` ou `truncated`) a une ligne `"abstained"` dans `labels.jsonl` (voir [judges/README.md](judges/README.md)) ;
une `error` du moteur n'en est pas une (elle peut être passagère : l'id reste sans ligne et `--resume` le redemande).
Pour les vérités, une abstention est un vote nul sur les trois champs : hors médiane, hors majorité et hors `spread`,
compté comme un `null` vers `undecidable`. `bench run` l'écrit dans la colonne `<juge>.abstained` et la liste dans
`run.json` ; un id sans aucune ligne reste refusé, jamais rempli. `bench agree` laisse hors de la paire un id où l'un
des deux juges s'est abstenu (colonne `abstained`).

## Dépendances

| Dépendance | Groupe | Raison |
|---|---|---|
| `pyarrow` | principal | lecture et écriture des fichiers parquet du jeu de prompts |
| `datasets` | `pool` | lecture des jeux publics dont le jeu de prompts est tiré |
| `huggingface_hub` | `pool` | accès en lecture seule aux jeux publics du Hub, sans jeton |
| `mlx-lm` | `judge` | exécution locale des juges sur Apple silicon |
| `pytest` | `dev` | tests, sans réseau et sans modèle |
| `ruff` | `dev` | lint et tri des imports |

## Licences

- Code : [Apache-2.0](LICENSE).
- Prompts d'autres jeux publics : les prompts de `data/pool/*`, de `data/prompts/trial-50.*` et des lots sont repris
  tels quels de Dolly (CC-BY-SA-3.0), d'OASST2 et de m-ArenaHard (Apache-2.0) ; chacun garde sa licence, donnée ligne
  par ligne dans la colonne `licence` de `pool.parquet` et de `<jeu>.meta.parquet`. Le détail est dans
  [data/README.md](data/README.md).
- [CC-BY-4.0](data/LICENSE) (`data/LICENSE`) : les prompts écrits à la main et leurs traductions, les fichiers de juges
  (`judges/`), les vérifications (`data/checks/`), la liste d'exclusion et les statistiques. Les traductions des prompts
  d'un jeu public sont des œuvres dérivées et gardent la licence de la ligne (colonne `licence` de `meta.parquet`).
