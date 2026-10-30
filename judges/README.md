# Les fichiers de juges

Un juge = un dossier `judges/<nom>/` de deux fichiers. `judges/example-chat/` en est un exemple complet, sur trois
prompts fictifs (`data/prompts/example.jsonl`).

## `meta.json`

Un objet JSON avec ces 13 clés, toutes obligatoires :

| clé | valeur |
|---|---|
| `judge` | nom libre du juge |
| `model` | identifiant exact du modèle |
| `provider` | `local`, `chat`, `api` ou `human` |
| `revision` | commit ou version si connue, sinon `"unknown"` |
| `date` | date du jugement, `AAAA-MM-JJ` |
| `temperature` | si connue, sinon `"unknown"` (toute valeur acceptée) |
| `thinking` | si connu, sinon `"unknown"` (toute valeur acceptée) |
| `instructions` | `v7-single` ou `v7-batch` (la consigne de `consigne/`) |
| `prompts_sha256` | empreinte du fichier de prompts jugé : le contenu de `data/prompts/<jeu>.sha256` |
| `author` | qui verse le fichier |
| `replayable` | booléen JSON : `true` seulement pour `local` à température 0 et révision figée |
| `licence` | exactement `CC-BY-4.0` : un fichier sous une autre licence ne peut pas être fusionné |
| `notes` | texte libre (pour une correction humaine : ce qui a été corrigé) |

Seules trois clés de plus sont tolérées, écrites par `bench judge` : `quantization`, `mlx_lm_version` et `max_tokens`
(un entier strictement positif). Toute autre clé est refusée.

## `labels.jsonl`

Une ligne par prompt du fichier de prompts désigné par `prompts_sha256`, un objet JSON avec exactement quatre champs :

```json
{"id": "bad9ea434b83cd45-eng", "complexity": "trivial", "task": "information_seeking", "needs_tools": false}
```

- `id` : une chaîne, l'`id` d'une ligne du fichier de prompts ; tous présents, aucun en trop, aucun en double.
- `complexity` : `trivial`, `low`, `medium`, `high`, `expert`, ou `null` (le JSON `null`, pas la chaîne) si le prompt ne
  peut pas être jugé.
- `task` : l'une des 19 tâches de la grille v7 (`information_seeking`, `reasoning`, ..., `chat`, `other`).
- `needs_tools` : `true` ou `false`, un vrai booléen JSON (ni `"true"`, ni `1`).

Aucun contrôle de qualité du contenu : un juge médiocre se voit dans son accord avec les autres.

## Un fichier accepté ne change plus

Un fichier accepté ne change plus. Une correction est un nouveau dossier `judges/<nom>-corrige-<auteur>/`, avec
`provider: human` et des `notes` qui disent ce qui a été corrigé. Les deux apparaissent dans le rapport.

Les dossiers `check-*` sont des juges du banc sur un sous-ensemble du jeu (les 90 prompts de base, dans une langue) ;
les dossiers `witness-*` (traductions témoins) et `presort-*` (pré-tri) n'en sont pas.

## Vérifier un dossier

```
uv run bench validate judges/<nom> [--prompts data/prompts]
```

`--prompts` désigne le dossier des fichiers de prompts (`data/prompts` par défaut). Dossier accepté : `ok` sur la sortie
standard, code de sortie 0. Dossier refusé : une faute par ligne sur la sortie d'erreur, de la forme
`labels.jsonl:LIGNE: ...` ou `meta.json: ...`, code de sortie 1.
