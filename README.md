# s1d1-bench

Banc ouvert pour mesurer à quel point un classifieur de prompts s'accorde avec des juges LLM locaux, sur la grille de
notation v7 (palier de complexité, score, tâche, `needs_tools`).

Les décisions citées `docs/NN` sont dans [`docs/`](docs/README.md).

## Ce que le banc mesure, et ce qu'il ne mesure pas

- **Mesuré.** L'accord des quatre sorties d'un classifieur avec la médiane des juges : palier de complexité, score,
  tâche et `needs_tools`, globalement, et pour le palier de complexité aussi par langue, par source, par origine et par
  palier de la vérité. Les témoins aléatoire et longueur du
  prompt donnent le plancher, l'accord de chaque juge avec les autres donne le plafond.
- **Non mesuré : la grille.** Si la grille elle-même correspond à « quel modèle suffit » relève d'un autre banc, celui
  des réponses, qui n'est pas traité ici.
- **Non mesuré : le routage.** Le mapping score vers modèle est l'affaire du déployeur.
- **Non mesuré : la distribution réelle.** Le jeu est construit par cases, `expert` sera la moins fournie ; les prompts
  agentiques et multi-tours sont absents, le jeu est fait de premiers prompts.
- **Non mesuré : les autres langues.** Seules les quatre langues choisies sont couvertes.

## Résultats (`base-v1`, 2026-10-30)

La phrase de docs/32 § 5, remplie avec les chiffres du rapport de `base-v1` (360 ids, 90 prompts de base) :

> Sur 360 ids, soit 90 prompts de base en 4 langues, `s1d1-full-int8` s'accorde avec la médiane de 3 juges à QWK 0,807
> [0,734 ; 0,852] ; le plafond inter-juges est 0,774 [0,718 ; 0,815] (chaque juge principal contre les deux autres, 8
> votes et non 12 : `base-v1-gemma4-cap512` 0,827, `base-v1-qwen3-cap512` 0,794, `base-v1-gptoss-cap512` 0,702) ; Laya
> english 0,405 [0,314 ; 0,490], Laya multilingual 0,119 [−0,040 ; 0,264] ; l'heuristique de LiteLLM 0,089 [0,026 ;
> 0,172] ; la longueur du prompt 0,543 [0,424 ; 0,647]. L'IC de `s1d1-full-int8` contient la moyenne du plafond ;
> `s1d1-full-int8` se distingue du plancher `length` selon la règle des 2 σ (écart apparié `s1d1-full-int8` −
> `length` : 0,264 [0,159 ; 0,383]).

![Accord de chaque candidat avec la médiane des juges principaux](report/base-v1/figures/qwk-bars.svg)

![QWK par langue](report/base-v1/figures/qwk-by-language.svg)

QWK contre `truth_all` (la médiane des trois juges principaux), IC à 95 % sur 1 000 tirages des 90 prompts de base ;
« palier exact » est la part des ids dont le palier est celui de `truth_all` :

| candidat | QWK [IC 95 %] | palier exact [IC 95 %] |
|---|---|---|
| s1d1 full int8 | 0,807 [0,734 ; 0,852] | 0,656 [0,569 ; 0,733] |
| s1d1 small int8 | 0,772 [0,700 ; 0,822] | 0,650 [0,561 ; 0,731] |
| plafond (moyenne des trois juges) | 0,774 [0,718 ; 0,815] | 0,600 [0,523 ; 0,671] |
| longueur (témoin `length`) | 0,543 [0,424 ; 0,647] | 0,394 [0,300 ; 0,483] |
| Laya english (`laya-english-tier`) | 0,405 [0,314 ; 0,490] | 0,311 [0,247 ; 0,386] |
| Laya multilingual (`laya-multilingual-tier`) | 0,119 [−0,040 ; 0,264] | 0,278 [0,203 ; 0,353] |
| LiteLLM heuristic | 0,089 [0,026 ; 0,172] | 0,342 [0,250 ; 0,436] |

Les juges sont l'instrument de mesure, pas des concurrents : aucun tableau ne range un candidat face à un juge, sauf le
plafond. Aucune phrase « meilleur que » sans la règle des 2 σ (l'IC de l'écart apparié exclut zéro et dépasse 2 σ,
σ = 0,005) : la règle ne sépare pas `s1d1-full-int8` de `s1d1-full-fp16`, de `s1d1-small-int8` ni de
`s1d1-compact-int8` (seul `tiny` en est séparé). Tout le reste est dans
[`report/base-v1/index.md`](report/base-v1/index.md) ; pour le régénérer, depuis la racine du dépôt :

```
uv run --group report bench report --results results/base-v1.parquet --meta data/prompts/base-v1.meta.parquet \
    --classifiers classifiers --out report/base-v1
```

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
- `bench classify` : fait classer les prompts par un classifieur et écrit un dossier `classifiers/<nom>` au format d'un juge,
  plus le score quand il est sur l'échelle v7. Six adaptateurs : trois témoins (`random`, `length`, `constant`) et trois
  systèmes à l'essai (`s1d1`, poids du Hub ; `laya` ; `litellm-heuristic`, le scorer à règles de LiteLLM, sans modèle), ce
  dernier et `laya` gardant leurs décisions brutes à part dans `decisions.jsonl` (palier nom à nom, sans `score` dans
  `labels.jsonl`).
- `bench run` : réunit les dossiers de juges et de classifieurs d'un jeu en un tableau, une ligne par id, avec les vérités
  des juges principaux (`results/<jeu>.parquet` et `<jeu>.run.json`).
- `bench report` : écrit le rapport d'un jeu (`report/<jeu>/index.md`, vingt-cinq tableaux CSV, sept figures SVG avec leur
  CSV) depuis `results/<jeu>.parquet` ; voir [Le rapport](#le-rapport).
- `bench profile` : mesure, pour chaque modèle d'un run, la latence par prompt à lot de 1 après chauffe (p50, p95,
  moyenne), la mémoire de crête du processus et le poids des fichiers de poids (`results/<jeu>.profile.json`) ; voir
  [Le profil](#le-profil).

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

Les 90 prompts de base anglais (`data/prompts/base-v1-eng`, export de `base-v1.select.txt` et des dix prompts écrits à
la main) sont traduits en français, russe et chinois simplifié par le modèle du pré-tri, Qwen3.8 27B 8 bits sans
réflexion, à température 0 ([docs/33](docs/33-translation-lot2.md)). `bench translate` traduit un fichier de prompts
dans une langue, un prompt par génération : la consigne (`consigne/translate-v1-<lang>.md`) est le message système, le
prompt est donné entre balises `<prompt>` comme une donnée.

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

## Les juges du rapport et les classifieurs

Le rapport de `base-v1` s'appuie sur cinq séries de juges et onze classifieurs, tous sur `data/prompts/base-v1.jsonl` (360
lignes, SHA-256 `e6a6d4c12ef8b95caea942337a3a1e16ec1622cb4e75c8ee0b473ef609a32490`). Les juges ont tourné le 2026-10-30
sur un Mac M4 Max 128 Go, avec `mlx-lm` 0.32.2, un modèle à la fois, température 0, `--max-tokens 1024`, consigne
`v7-single`, révisions de [judges/MODELS.md](judges/MODELS.md).

### Les cinq juges

| dossier | réglage | durée | rôle |
|---|---|---|---|
| `judges/base-v1-gemma4-cap512` | Gemma 4 31B, raisonnement plafonné à 512 | 4 h 28 min (4,48 h) | médiane principale |
| `judges/base-v1-qwen3-cap512` | Qwen3.8 27B, raisonnement plafonné à 512 | 3 h 02 min (3,04 h) | médiane principale |
| `judges/base-v1-gptoss-cap512` | gpt-oss-20b, raisonnement plafonné à 512 | 11 min 36 s (0,19 h) | médiane principale |
| `judges/base-v1-gemma4-off` | Gemma 4 31B, sans raisonnement | 1 h 18 min (1,30 h) | effet de la réflexion, hors médiane |
| `judges/base-v1-qwen3-off` | Qwen3.8 27B, sans raisonnement | 56 min 41 s (0,94 h) | effet de la réflexion, hors médiane |

Les durées vont du début à la fin de chaque série, soit 9 h 57 min 38 s au total (le plan en prévoyait 8,4 h) ; celle de
gpt-oss ne compte pas les 9 s de la reprise qui a écrit l'abstention. La première série (`qwen3-off`) a tourné seule sur
la machine ; les autres ont tourné pendant que des sous-agents de relecture et des essais légers s'en servaient aussi :
les durées ne se comparent donc pas à la seconde près. Les trois juges `cap512` forment la médiane principale : Gemma 4
est le réglage du juge de l'entraînement, et [docs/32](docs/32-open-bench-kit.md) demande au moins trois juges locaux,
gpt-oss étant le troisième modèle qui lit le JSON sur les 50 prompts de l'essai
([docs/34](docs/34-bench-run-report-lot3.md) décision 2). Les deux dossiers `off` mesurent l'effet « réflexion » sur les
360 prompts ; ils figurent dans le tableau des juges mais pas dans la médiane principale, parce que deux réglages d'un
même modèle ne sont pas deux juges indépendants.

**Le cas d'abstention vient en premier.** `judges/base-v1-gptoss-cap512` compte 359 étiquettes et une abstention : gpt-oss
refuse l'id `3ce175ed3ef6b933-zho` (le prompt chinois de synthèse de l'acide chlorhydrique) et rend un JSON illisible
(`json_error`). Ce n'est pas une étiquette inventée : le dossier porte, pour cet id, une ligne `"abstained": "json_error"`
aux trois champs `null` (la ligne a été écrite par un simple `--resume` avec les mêmes réglages, après la règle des
abstentions), et `bench validate` affiche `ok (1 abstention)`. Ce qu'une abstention est et vaut pour les vérités est dit dans
[Les vérités](#les-vérités) et dans [judges/README.md](judges/README.md) ; la section n'y revient pas. Dans les accords
ci-dessous, la colonne `abstained` compte les ids laissés hors de la paire.

### Les treize dossiers de classifieurs (onze dans les résultats)

| dossier | ce que c'est | révision | durée |
|---|---|---|---|
| `classifiers/random` | témoin : tirage uniforme, graine 0, 10 tirages (`labels.jsonl` = tirage 0, `draws.jsonl` = les dix) | graine 0 | n/a |
| `classifiers/length` | témoin : déciles de la longueur en caractères, par langue, correspondance 2/3/3/1/1 | sans objet | n/a |
| `classifiers/constant-low` | témoin : `low` partout, score 0,25 | sans objet | n/a |
| `classifiers/s1d1-full-fp16` | `scombat/s1d1-router`, PyTorch fp16, mps | tête du Hub `7fb9a4e` | 7,9 s |
| `classifiers/s1d1-full-int8` | `scombat/s1d1-router-6src-full-onnx-int8`, ONNX int8, CPU | carte `068d0d2` | 35,6 s |
| `classifiers/s1d1-compact-int8` | `scombat/s1d1-router-6src-compact-onnx-int8`, ONNX int8, CPU | carte `7e87870` | 28,5 s |
| `classifiers/s1d1-small-int8` | `scombat/s1d1-router-6src-small-onnx-int8`, ONNX int8, CPU | carte `33c7f34` | 12,2 s |
| `classifiers/s1d1-tiny-int8` | `scombat/s1d1-router-6src-tiny-onnx-int8`, ONNX int8, CPU | tête du Hub `dac13b5` | 6,7 s |
| `classifiers/laya-english` | ne compte plus (voir plus bas) : Laya 0.4.15, point de contrôle english, mps, ancre la plus proche | `convaiinnovations/laya@7b928d82…` | 13,1 s |
| `classifiers/laya-multilingual` | ne compte plus (voir plus bas) : Laya 0.4.15, point de contrôle multilingual, mps, ancre la plus proche | `convaiinnovations/laya@7b928d82…` | 9,0 s |
| `classifiers/laya-english-tier` | Laya 0.4.15, point de contrôle english, mps, palier le plus probable nom à nom | `convaiinnovations/laya@7b928d82…` | ≈ 20 s (non chronométré) |
| `classifiers/laya-multilingual-tier` | Laya 0.4.15, point de contrôle multilingual, mps, palier le plus probable nom à nom | `convaiinnovations/laya@7b928d82…` | ≈ 20 s (non chronométré) |
| `classifiers/litellm-heuristic` | scorer heuristique intégré de LiteLLM (`classifier_type: heuristic`, configuration par défaut), code copié tel quel au tag v1.106.6, sans modèle | tag `v1.106.6`, commit `61c6ab78…` | moins d'une seconde |

Les durées des classifieurs sont celles du classement des 360 prompts, avec le petit téléchargement du Hub, les poids étant
déjà en cache. Trois révisions de s1d1 (`small`, `compact`, `full` int8) sont épinglées à la révision de la carte du dépôt.
Pour `tiny` en int8 et pour le modèle fp16, la carte ne donne pas de révision : le banc prend la tête du Hub au moment de
l'exécution, après avoir vérifié (lecture seule, sans jeton) que les fichiers de poids n'ont pas changé depuis le dernier
commit qui les a modifiés ; la vérification est écrite dans le champ `notes` du `meta.json` de chaque dossier. Pour le
fp16, `config.json` et `s1d1_router.py` ont changé après l'ajout des poids : « sans changement » vaut pour les fichiers de poids, et la tête du dépôt
fournit le `config.json` et le routeur utilisés. Laya est épinglé à `convaiinnovations/laya@7b928d82…` (le hachage entier est dans `meta.json`).

**Le scorer de LiteLLM.** `classifiers/litellm-heuristic` n'a ni modèle ni réseau : le banc embarque le code du scorer
heuristique de LiteLLM, copié tel quel au tag `v1.106.6` (commit `61c6ab78393ef45683dd7d963d2a315bd845e20b`) dans
`src/s1d1_bench/vendor/litellm_heuristic/`, sans installer le paquet `litellm`. Le fichier `NOTICE.md` à côté du code dit
d'où il vient, quels blocs sont repris et comment ; la licence de ce code est celle de LiteLLM, MIT (`LICENSE`, copié
avec lui). Chaque prompt est classé seul, sans consigne système, avec la configuration par défaut de LiteLLM ; les
paliers se traduisent nom à nom (`SIMPLE` en `low`, `MEDIUM` en `medium`, `COMPLEX` en `high`, `REASONING` en `expert`),
et `task` et `needs_tools` restent `null`. Son score n'est pas sur l'échelle v7 (il peut être négatif) : `labels.jsonl` n'a
donc pas de `score`, et la décision brute du scorer (palier, score, signaux, cause) est dans `decisions.jsonl`, à côté de
`labels.jsonl`. `bench run` la lit dans les colonnes `litellm-heuristic.raw_score` et `litellm-heuristic.raw_tier`, et
marque `"raw_decisions": true` dans l'entrée du classifieur de `run.json`. Le score brut ne sert qu'aux mesures de rang.

**Révisions du Hub pour les dépôts int8.** Pour `small`, `compact` et `full`, la tête actuelle du Hub n'est pas la révision de
la carte : les commits qui les séparent ne touchent que des métadonnées (`README.md`, `eval.json`, `footprint.json`,
`variant.json`, où seuls `sigma` et sa date changent) ; les fichiers `*.onnx` et `*.npy` ont le même SHA-256 LFS à la tête
et à la révision de la carte. Les dossiers gardent la révision de la carte, la tête et cette vérification étant
dans leurs `notes`.

### Rejouer

Toutes les commandes se lancent depuis la racine du dépôt, les modèles de juges étant déjà dans `~/mlx-models/` (voir
l'étape 4 de « Reproduire l'essai »). Un dossier existant n'est jamais réécrit : pour refaire une série, la sortir sous un
autre nom.

- **Un juge** (Apple silicon, groupe `judge`) :

  ```
  uv run --group judge bench judge --model mlx-community/gemma-4-31b-it-8bit --prompts data/prompts/base-v1.jsonl \
      --out judges/base-v1-gemma4-cap512 --author <nom> --thinking cap:512 --max-tokens 1024
  ```

  À répéter pour `mlx-community/Qwen3.8-27B-8bit` (`qwen3`) et `mlx-community/gpt-oss-20b-MXFP4-Q8` (`gptoss`) en
  `cap:512`, et pour Gemma et Qwen en `--thinking off` (`base-v1-gemma4-off`, `base-v1-qwen3-off`). Le `meta.json` de
  chaque dossier dit la révision, le réglage de la réflexion et le rôle du dossier.
- **Les témoins**, sans groupe particulier :

  ```
  uv run bench classify random --seed 0 --draws 10 --prompts data/prompts/base-v1.jsonl --out classifiers/random --author <nom>
  uv run bench classify length --prompts data/prompts/base-v1.jsonl --out classifiers/length --author <nom>
  uv run bench classify constant --level low --prompts data/prompts/base-v1.jsonl --out classifiers/constant-low --author <nom>
  ```
- **s1d1 en ONNX int8** (CPU, groupe `s1d1`), avec la révision de la carte :

  ```
  uv run --group s1d1 bench classify s1d1 --model scombat/s1d1-router-6src-small-onnx-int8 --revision 33c7f34a30af04735390264824d6bfa8546abb7a \
      --backend onnx --prompts data/prompts/base-v1.jsonl --out classifiers/s1d1-small-int8 --author <nom>
  ```

  `compact` (`7e87870132a19d12b32dbad1f8233801b8145b9f`) et `full` (`068d0d20d5901c2db6934c89fde48b87c973fed8`) de la même
  façon ; `tiny` (`dac13b5c0356864ea4501fa151d74dcd8b3ea22a`, la tête du Hub) aussi, avec son identifiant
  `scombat/s1d1-router-6src-tiny-onnx-int8`.
- **s1d1 en fp16** (PyTorch, groupes `s1d1-torch` et `s1d1`) :

  ```
  uv run --group s1d1 --group s1d1-torch bench classify s1d1 --model scombat/s1d1-router \
      --revision 7fb9a4e3d03e3d942a98648268ff76c6152c18d6 --backend torch --prompts data/prompts/base-v1.jsonl \
      --out classifiers/s1d1-full-fp16 --author <nom>
  ```
- **Laya** (groupe `laya`, à part) :

  ```
  uv run --group laya bench classify laya --laya-model english --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
      --prompts data/prompts/base-v1.jsonl --out classifiers/laya-english-tier --author <nom>
  ```

  `--laya-model multilingual` et `--out classifiers/laya-multilingual-tier` pour l'autre point de contrôle.

  **Deux dossiers Laya par point de contrôle, et lesquels comptent.** Laya répond à la question à quatre paliers de
  LiteLLM (`SIMPLE`, `MEDIUM`, `COMPLEX`, `REASONING`), la même que celle du scorer de LiteLLM ci-dessous, et les deux
  candidats se lisent avec la même table, nom à nom : `SIMPLE` → low, `MEDIUM` → medium, `COMPLEX` → high, `REASONING` →
  expert, jamais trivial. Les dossiers **`laya-english-tier` et `laya-multilingual-tier` sont ceux qui comptent** : le palier
  est le plus probable des quatre (égalité : le nom le plus bas), `labels.jsonl` n'a pas de `score`, et `decisions.jsonl` garde
  par id le palier de Laya, le niveau attendu `Σ i·pᵢ` sur 0..3 (écrit `score`, il ne sert qu'aux métriques de rang) et les
  quatre probabilités telles quelles. Les dossiers **`laya-english` et `laya-multilingual`** de la première PR convertissaient
  le niveau attendu divisé par 3 en palier par l'ancre la plus proche (SIMPLE devenait trivial, MEDIUM low, COMPLEX high) :
  une autre table que celle du scorer de LiteLLM. Ils restent dans le dépôt, parce qu'un dossier de classifieur ne se
  modifie jamais, mais `bench run` ne les charge plus : ils ne sont ni dans les résultats ni dans les chiffres du rapport,
  qui dit seulement pourquoi.
- **Le scorer de LiteLLM** (sans modèle, sans groupe, sans réseau) :

  ```
  uv run bench classify litellm-heuristic --prompts data/prompts/base-v1.jsonl --out classifiers/litellm-heuristic --author <nom>
  ```
- **Le tableau** :

  ```
  uv run bench run --prompts data/prompts/base-v1.jsonl --judges 'judges/base-v1-*' \
      --main base-v1-gemma4-cap512,base-v1-qwen3-cap512,base-v1-gptoss-cap512 \
      --classifiers 'classifiers/s1d1-*' classifiers/laya-english-tier classifiers/laya-multilingual-tier \
      classifiers/litellm-heuristic classifiers/random classifiers/length classifiers/constant-low \
      --out results/base-v1.parquet
  ```

  `--judges` et `--classifiers` prennent plusieurs valeurs à la suite (des globs entre apostrophes, que `bench run`
  développe lui-même), ou l'option répétée.

  Les classifieurs sont listés un à un : `'classifiers/*'` chargerait aussi `laya-english` et `laya-multilingual`, qui ne
  comptent plus (voir Laya plus haut).
- **Vérifier** les dossiers déposés, comme le fait la CI (`.github/workflows/ci.yml`) :

  ```
  for d in judges/*/ classifiers/*/; do uv run bench validate "$d"; done
  ```

Attention : `uv sync --group <g>` rend l'environnement exact et retire les groupes qu'il ne nomme pas ; synchroniser le
groupe `judge` puis le groupe `laya` désinstalle le premier. Pour une commande, `uv run --group <g> ...` ; pour un
environnement complet, `uv sync --all-groups` (qui comprend `mlx-lm`, donc Apple silicon seulement).

### Les règles de vérité

Les vérités (médiane toutes langues sur 12 votes, médiane par langue sur 3 votes, règle R5, indécidables, vote majoritaire
de `task` et de `needs_tools`, abstentions) sont celles de [Les vérités](#les-vérités) ; elles ne sont pas répétées
ici. Le comptage de `bench run` sur `base-v1`, avec les trois juges principaux : 1 abstention, aucun id indécidable ;
33 ids sur 360 ont un désaccord de langue (24 des 90 prompts de base), écart moyen 0,675 (écart 0 pour 140 ids, 1 pour 197,
2 pour 23) ; `truth_all.complexity` donne trivial 68, low 116, medium 116, high 52, expert 8, et `truth_all.needs_tools`
est vrai pour 52 ids. Une médiane basse simple, à la place de la règle R5, changerait `truth_all` pour 0 des 90 clés de base, et
`truth_all_nogemma` pour 1.

### Ce que `results/` est, et n'est pas

`results/base-v1.parquet` (360 lignes, 92 colonnes) et `results/base-v1.run.json` (les dossiers chargés, leurs révisions,
les abstentions, les règles) sont **dérivés** des dossiers `judges/` et `classifiers/` : ils se régénèrent à chaque
`bench run` ; les colonnes d'un classifieur sont `<nom>.complexity`, `<nom>.task`, `<nom>.needs_tools`, `<nom>.score` quand son
`labels.jsonl` en a un et, quand il a un `decisions.jsonl`, `<nom>.raw_score` et `<nom>.raw_tier` (`litellm-heuristic` a
ces deux-là et pas de `score` : cinq colonnes ; `laya-english-tier` et `laya-multilingual-tier` aussi). Ce n'est pas un enregistrement immuable, à la différence des dossiers de
juges et de classifieurs, qu'on ajoute et qu'on ne réécrit pas : ce qui se rejoue, ce sont les entrées. Ce n'est pas non
plus un rapport : aucune métrique, aucune figure ne s'y trouve, elles sont dans `report/` (voir [Le rapport](#le-rapport)).

### Les comptes des juges

Répartition des paliers de chaque juge (`needs_tools` : nombre d'ids étiquetés vrais ; `null` : paliers laissés sans
étiquette, l'abstention de gpt-oss et un palier `null` de Gemma `cap512`, sortie du juge) :

| juge | trivial | low | medium | high | expert | null | needs_tools vrai |
|---|---|---|---|---|---|---|---|
| base-v1-gemma4-cap512 | 51 | 135 | 105 | 58 | 10 | 1 | 61 |
| base-v1-gemma4-off | 59 | 130 | 105 | 43 | 23 | 0 | 57 |
| base-v1-gptoss-cap512 | 81 | 122 | 143 | 13 | 0 | 1 | 40 |
| base-v1-qwen3-cap512 | 85 | 79 | 106 | 71 | 19 | 0 | 59 |
| base-v1-qwen3-off | 70 | 85 | 92 | 91 | 22 | 0 | 69 |

Accord deux à deux (`bench agree judges/base-v1-*`) :

| juge A | juge B | n | abstained | palier exact | palier ±1 | task | needs_tools |
|---|---|---|---|---|---|---|---|
| base-v1-gemma4-cap512 | base-v1-gemma4-off | 360 | 0 | 81,4 % | 99,4 % | 85,6 % | 97,2 % |
| base-v1-gemma4-cap512 | base-v1-gptoss-cap512 | 359 | 1 | 56,3 % | 96,4 % | 56,5 % | 91,4 % |
| base-v1-gemma4-cap512 | base-v1-qwen3-cap512 | 360 | 0 | 70,3 % | 99,7 % | 83,1 % | 93,9 % |
| base-v1-gemma4-cap512 | base-v1-qwen3-off | 360 | 0 | 65,8 % | 99,7 % | 80,3 % | 94,4 % |
| base-v1-gemma4-off | base-v1-gptoss-cap512 | 359 | 1 | 55,4 % | 94,7 % | 49,3 % | 91,9 % |
| base-v1-gemma4-off | base-v1-qwen3-cap512 | 360 | 0 | 72,2 % | 99,7 % | 80,3 % | 95,6 % |
| base-v1-gemma4-off | base-v1-qwen3-off | 360 | 0 | 67,8 % | 99,7 % | 83,3 % | 95,0 % |
| base-v1-gptoss-cap512 | base-v1-qwen3-cap512 | 359 | 1 | 47,1 % | 93,9 % | 59,6 % | 91,9 % |
| base-v1-gptoss-cap512 | base-v1-qwen3-off | 359 | 1 | 46,5 % | 91,4 % | 53,5 % | 90,8 % |
| base-v1-qwen3-cap512 | base-v1-qwen3-off | 360 | 0 | 77,8 % | 100,0 % | 84,7 % | 95,6 % |

Ces deux tableaux sont des comptes, donnés une fois ici, sans intervalle ni métrique d'accord ; les QWK, leurs intervalles
de confiance et ce qu'ils disent sont dans [Le rapport](#le-rapport).

## Le rapport

`report/base-v1/` est écrit par `bench report` à partir de `results/base-v1.parquet` ; il est dérivé comme `results/`,
et se régénère (même entrée, mêmes octets : le rapport est déterministe, SVG compris). Le refaire :

```
uv run --group report bench report --results results/base-v1.parquet --meta data/prompts/base-v1.meta.parquet \
    --classifiers classifiers --out report/base-v1
```

`--classifiers` est la racine des dossiers : la bande du témoin aléatoire se lit dans `classifiers/random/draws.jsonl`
(dix tirages), `results/base-v1.run.json` ne donnant que les noms. Une quinzaine de secondes. La commande refuse de
tourner (code 1, un message) si un dossier de classifieur nommé par `run.json` manque à la racine, si `run.json` n'a pas
l'une des clés `classifiers`, `judges`, `main` ou la table méta l'une des colonnes `source`, `origin`, si le gabarit
`report/TEMPLATE.md` est introuvable, ou si `matplotlib` (groupe `report`) n'est pas installé.

**Ce qu'il contient.** `index.md` (lu de haut en bas : les deux prédictions de docs/34 § 7.4 d'abord, celle qui ne se
vérifie pas en tête ; la phrase de docs/32 § 5 avec ses chiffres ; les mises en garde de docs/32 § 2 ; l'accord de chaque
candidat et de chaque juge avec `truth_all` puis `truth_all_nogemma` ; le plafond, le plancher, les écarts appariés ; par
langue, par source, par origine, par palier ; `task` et `needs_tools` ; les rangs et la calibration du score ; ce qui
manque), `tables/` (vingt-cinq CSV) et `figures/` (sept SVG, chacun avec le CSV de ses données). Le gabarit
`report/TEMPLATE.md` porte les mises en garde et la phrase à trous ; le code remplit les trous avec les chiffres tels
qu'ils sortent, et les deux clauses finales de la phrase suivent des règles écrites dans `src/s1d1_bench/report_index.py`
et citées dans le rapport. Les intervalles sont à 95 %, par rééchantillonnage des 90 prompts de base (1 000 tirages,
graine 0, les quatre versions suivent leur prompt de base).

**Ce qu'il dit.** Sur 360 ids (90 prompts de base en quatre langues), `s1d1-full-int8` s'accorde avec la médiane des trois
juges principaux à QWK 0,807 [0,734 ; 0,852] et `s1d1-full-fp16` à 0,814 [0,738 ; 0,865] ; le plafond (chaque juge
principal contre les deux autres, 8 votes) vaut 0,774 [0,718 ; 0,815] en moyenne ; le plancher informé, la longueur du
prompt, 0,543 [0,424 ; 0,647], et le hasard −0,028 à 0,049 sur dix tirages. Laya, lu nom à nom, est à 0,405 [0,314 ; 0,490]
(english) et 0,119 [−0,040 ; 0,264] (multilingual) ; l'heuristique de LiteLLM à 0,089 [0,026 ; 0,172]. **La prédiction 2 de
docs/34 § 7.4 (LiteLLM entre `length` et s1d1) ne se vérifie pas** : LiteLLM est au-dessous de `length`, et sa prédiction
de palier est `low` pour 93 % des ids, son exactitude (0,342) restant à côté de celle de `constant-low` (0,322). La
prédiction 1 (son QWK tombe en français, russe et chinois) se vérifie sur les valeurs ponctuelles, sans que les IC se
séparent. Chaque chiffre est dans un tableau de `report/base-v1/tables/`, cité dans `index.md`.

**Ce qu'il ne prouve pas.** Il ne sépare pas `s1d1-full-int8` de `s1d1-full-fp16`, de `small` ni de `compact` : les
écarts appariés de `fp16` et de `small` ont un IC qui contient zéro, celui de `compact` (0,048 [0,009 ; 0,089]) exclut
zéro mais sa borne basse n'est pas au-delà de 2 σ = 0,010 (σ = 0,005, [docs/18](docs/README.md), lecture stricte écrite
dans le rapport). Les trois juges principaux sont dans la vérité : leur accord avec elle compte leurs propres votes, et
le chiffre qui les compare est le plafond. Gemma 4 a produit les étiquettes d'entraînement de s1d1 (docs/32 § 4.4) et
est dans `truth_all` : le rapport donne l'accord contre la médiane sans Gemma (`truth_all_nogemma`, tableau « Sans Gemma
4 ») ; `s1d1-full-int8` y passe de 0,807 à 0,785 (`fp16` de 0,814 à 0,794) ; parmi les candidats qui n'ont pas vu Gemma,
`length` passe de 0,543 à 0,478 et `laya-english-tier` de 0,405 à 0,347, tandis que `litellm-heuristic` (0,089 → 0,106)
et `laya-multilingual-tier` (0,119 → 0,115), proches du plancher, ne baissent guère. Ces écarts se lisent sur les
valeurs ponctuelles, sans IC (aucun écart apparié entre vérités n'est calculé). Sans Gemma, la vérité est la médiane des
8 votes de deux juges, arrondie vers le bas : `expert` disparaît (8 ids → 0), `high` passe de 52 à 28 ids, 20 des 90
prompts de base changent de palier (18 vers le bas), et un candidat qui ne prédit jamais `expert`, comme s1d1, peut y
perdre moins. Les deux juges restants s'accordent avec Gemma (QWK 0,879 pour Qwen 3, 0,707 pour gpt-oss, tableau des
juges). Le tableau « Sans Gemma 4 » ne dit donc pas si s1d1 hérite des biais de Gemma. Le rapport ne mesure pas la
grille, le routage, la distribution réelle des prompts, les prompts agentiques ou multi-tours, ni les langues autres que
les quatre ; trois analyses de [docs/32](docs/32-open-bench-kit.md) ne sont pas produites dans ce lot (la part des
prompts notés 0,70 à 0,80 que les juges mettent en `high` ou au-dessus, les confusions entre tâches voisines, la matrice
de confusion normalisée de la tâche), et `expert` n'a que 8 ids sur 360 (2 prompts de base) : tout ce qui touche ce
palier est dit sur très peu de prompts. Les 360 ids ne sont pas 360 prompts indépendants : ce sont 90 prompts de base en
quatre versions, et les IC en tiennent compte.

### Le profil

`results/base-v1.profile.json` est écrit par `bench profile` à partir de `results/base-v1.run.json`. Il se refait sur
le Mac, avec les poids déjà présents en local : la commande pose `HF_HUB_OFFLINE=1`, rien n'est téléchargé, et un modèle
absent du cache échoue au lieu de se télécharger.

```
uv run --group s1d1 --group s1d1-torch --group laya bench profile --prompts data/prompts/base-v1.jsonl \
    --run results/base-v1.run.json --judges-dir judges --models-dir ~/mlx-models --out results/base-v1.profile.json
```

**Ce qu'il mesure.** Pour chaque classifieur du run dont le `meta.json` nomme un modèle (s1d1 ×5, Laya ×2,
l'heuristique de LiteLLM), la commande recharge le modèle par son adaptateur, aux mêmes `model`, `revision`, backend et
checkpoint (la ligne « Replay » de `meta.json`), dans un processus Python neuf ; elle le chauffe sur 10 appels, puis
chronomètre un appel par prompt du fichier, un prompt par appel (lot de 1, `time.perf_counter_ns`, 360 prompts) : p50,
p95 et moyenne en millisecondes. La mémoire de crête est le `ru_maxrss` de ce processus, lu après les appels
chronométrés ; un processus par modèle, pour que la crête de l'un ne soit pas celle d'un autre. Le poids est la taille
sur disque des fichiers de poids de la révision épinglée dans le cache local du Hub (le `.onnx` int8 des variantes int8,
le `.safetensors` fp16 de `s1d1-full-fp16`, le fichier de chaque tête pour Laya ; 0 pour LiteLLM), en Mo de 10⁶ octets.
Pour les juges, la latence vient de leur `log.jsonl` (`wall_s` des appels finis `ok`, un prompt par appel par
construction) et le poids des fichiers de `~/mlx-models/<org>--<nom>`. Les appareils sont ceux que les adaptateurs
rapportent : s1d1 int8 et LiteLLM sur CPU, s1d1 fp16 et Laya sur le GPU du Mac (MPS), les juges sur le GPU par MLX. Les
témoins ne sont pas des modèles : ils sont listés dans `meta.skipped`, comme le serait un modèle qui n'a pas pu se
charger (le fichier est alors écrit sans lui et la commande sort en code 1). `--only nom,...` restreint la mesure à des
noms du run, les autres allant dans `meta.skipped` ; un nom hors du run est refusé. `bench validate` ne lit pas ce
fichier.

**Ce qu'il ne dit pas.** La latence d'un juge comprend la lecture de la consigne v7 (environ 1 900 jetons) et la
génération de sa réponse, raisonnement compris : elle dit ce que coûte un juge employé comme routeur, pas la vitesse du
modèle sur le seul prompt. La mémoire de crête comprend l'interpréteur, les bibliothèques (onnxruntime, torch,
tokenizers) et le chargement du modèle : ce n'est pas la taille du modèle seul ; sur MPS, la mémoire tenue par le pilote
du GPU peut y manquer ; pour les juges, elle n'est pas mesurable après coup (`null`). Les latences bougent d'une
exécution à l'autre : le fichier est une mesure, pas un calcul, et ne se refait pas à l'octet près. Il a été mesuré une
fois, sur un M4 Max au repos, en une minute (58 s pour les huit classifieurs et les cinq juges).

## Dépendances

| Dépendance | Groupe | Raison |
|---|---|---|
| `pyarrow` | principal | lecture et écriture des fichiers parquet du jeu de prompts |
| `datasets` | `pool` | lecture des jeux publics dont le jeu de prompts est tiré |
| `huggingface_hub` | `pool` | accès en lecture seule aux jeux publics du Hub, sans jeton |
| `mlx-lm` | `judge` | exécution locale des juges sur Apple silicon |
| `onnxruntime`, `tokenizers`, `numpy`, `huggingface_hub` | `s1d1` | classifieur s1d1 en ONNX int8 (CPU) |
| `torch`, `transformers` | `s1d1-torch` | classifieur s1d1 en PyTorch fp16 |
| `laya` (0.4.15) | `laya` | classifieur Laya |
| `matplotlib` (et `numpy`, qu'il apporte) | `report` | figures SVG de `bench report` et statistiques de `metrics.py` |
| aucune | aucun | classifieur `litellm-heuristic` : code de LiteLLM copié dans le dépôt, sans le paquet `litellm` |
| `pytest` | `dev` | tests, sans réseau et sans modèle |
| `ruff` | `dev` | lint et tri des imports |

## Licences

- Code : [Apache-2.0](LICENSE), **sauf** `src/s1d1_bench/vendor/litellm_heuristic/`, copie mot pour mot du scorer
  heuristique de LiteLLM au tag v1.106.6, sous licence MIT (licence et notice des changements dans le dossier).
- Prompts d'autres jeux publics : les prompts de `data/pool/*`, de `data/prompts/trial-50.*` et des lots sont repris
  tels quels de Dolly (CC-BY-SA-3.0), d'OASST2 et de m-ArenaHard (Apache-2.0) ; chacun garde sa licence, donnée ligne
  par ligne dans la colonne `licence` de `pool.parquet` et de `<jeu>.meta.parquet`. Le détail est dans
  [data/README.md](data/README.md).
- Arena-Hard : les prompts anglais d'Arena-Hard viennent d'`arena-hard-auto` v0.1 (lmarena, Apache-2.0, prompts
  d'utilisateurs de Chatbot Arena) et leurs traductions de `CohereLabs/m-ArenaHard` (Apache-2.0) ; les conditions de la
  collecte amont auprès des utilisateurs de Chatbot Arena ne sont pas vérifiables en aval.
- [CC-BY-4.0](data/LICENSE) (`data/LICENSE`) : les prompts écrits à la main et leurs traductions, les fichiers de juges
  (`judges/`), les fichiers de classifieurs (`classifiers/`), les résultats (`results/`), le rapport (`report/<jeu>/`),
  la rubrique `src/s1d1_bench/rubric/judge-rubric-v7.txt`, les consignes `consigne/`, les vérifications
  (`data/checks/`), les documents copiés dans `docs/` (voir [docs/README.md](docs/README.md)), la liste d'exclusion
  et les statistiques. Les traductions des prompts d'un jeu public sont des
  œuvres dérivées et gardent la licence de la ligne (colonne `licence` de `meta.parquet`).
