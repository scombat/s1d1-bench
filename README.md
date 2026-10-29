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

Installation : `uv sync --group pool --group dev` (ajouter `--group judge` sur Apple silicon pour faire tourner un
juge), puis `uv run bench --help`.

## Reproduire l'essai

L'essai fait juger 50 prompts par trois modèles locaux, chacun avec le raisonnement coupé (`off`) et plafonné à 512
jetons (`cap:512`), soit six dossiers `judges/trial-*`. Les résultats et les durées sont dans
[judges/TRIAL.md](judges/TRIAL.md) : c'est un essai de l'outillage sur 50 prompts, pas un résultat du banc. Les trois
modèles, leurs révisions et la façon dont `bench judge` coupe ou plafonne leur raisonnement sont dans
[judges/MODELS.md](judges/MODELS.md) ; le format des dossiers de juges est dans [judges/README.md](judges/README.md).

Toutes les commandes se lancent depuis la racine du dépôt. Les étapes 4 et 5 demandent un Mac Apple silicon et
`uv sync --group judge` (ou `uv run --group judge`), un seul modèle tourne à la fois.

1. **Liste d'exclusion** (`data/exclusions/s1d1-seen-keys.txt`). Elle est produite par le dépôt de recherche privé
   s1d1-model-router, sur sa branche `seen-keys` (`pipeline/13_seen_keys.py`). La commande se lance dans ce dépôt de
   recherche ; `--out` est le chemin de la liste dans ce dépôt-ci (écrit ici en relatif, à adapter) :

   ```
   uv run python -I pipeline/13_seen_keys.py --inputs 'dataset/interim/*.parquet' 'dataset/final/*.parquet' \
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

3. **Les 50 prompts de l'essai.** Le tirage (graine 0 ; tous les candidats OASST2, puis 24 Dolly et 23 Arena-Hard)
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
   quota = {"oasst2": None, "dolly": 24, "arena_hard": 23}  # None: all of them
   keys = []
   for source, n in quota.items():
       candidates = sorted(by_source[source])
       keys += candidates if n is None else rng.sample(candidates, n)
   assert len(keys) == len(set(keys)) == 50
   open("data/prompts/trial-50.select.txt", "w", encoding="utf-8").write("\n".join(keys) + "\n")
   PY

   uv run bench export --pool data/pool/pool.parquet --select data/prompts/trial-50.select.txt --langs eng \
       --name trial-50 --seed 0
   ```

4. **Les modèles**, dans `~/mlx-models/<org>--<nom>` (jamais dans un dépôt). `bench judge` affiche la commande exacte
   s'il en manque un ; les identifiants et les révisions à utiliser sont dans [judges/MODELS.md](judges/MODELS.md) :

   ```
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download <org>/<nom> --revision <révision> \
       --local-dir ~/mlx-models/<org>--<nom>
   ```

   à faire pour `mlx-community/gemma-4-31b-it-8bit`, `mlx-community/Qwen3.8-27B-8bit` et
   `mlx-community/gpt-oss-20b-MXFP4-Q8`.

5. **Les six juges**, l'un après l'autre (provider `local`, MLX, Apple silicon seulement). Pour chaque famille,
   `--thinking off` puis `--thinking cap:512` ; `--author` reçoit votre nom :

   ```
   export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
   uv run --group judge bench judge --model mlx-community/gpt-oss-20b-MXFP4-Q8 --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-gptoss-off --author <nom> --thinking off --max-tokens 1024
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
- Données (`data/`) et fichiers de juges (`judges/`) : [CC-BY-4.0](data/LICENSE), le fichier `data/LICENSE` couvre les
  deux dossiers.
