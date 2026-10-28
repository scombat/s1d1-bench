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
- `bench agree` : calcule l'accord d'un classifieur avec les juges.

Installation : `uv sync --group pool --group dev` (ajouter `--group judge` sur Apple silicon pour faire tourner un
juge), puis `uv run bench --help`.

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
- Données (`data/`) : [CC-BY-4.0](data/LICENSE).
