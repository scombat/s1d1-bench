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

### Les abstentions

Quand le modèle ne rend pas d'étiquette lisible après la relance (état final `json_error` ou `truncated` dans
`log.jsonl`, par exemple un refus), `bench judge` écrit une ligne d'abstention plutôt que rien :

```json
{"id": "3ce175ed3ef6b933-zho", "complexity": null, "task": null, "needs_tools": null, "abstained": "json_error"}
```

Un refus est une sortie du juge, comme un palier `null` ; ce qui est interdit, c'est d'inventer une étiquette, pas de
consigner une non-réponse. Une réponse `ok` n'est jamais une abstention. Une `error` non plus : c'est une exception du
moteur, pas une sortie du modèle, et elle peut être passagère ; l'id reste sans ligne et `--resume` le redemande.
`bench validate` accepte une ligne `abstained` seulement si ses trois champs sont `null`, si l'état est l'un des deux
et, quand `log.jsonl` est à côté, si la dernière ligne de cet id dans le journal porte le même état ; un classifieur
(`provider: system`) ne s'abstient jamais. Le résumé devient `ok (1 abstention)`, `ok (N abstentions)`. `--resume` ne
redemande que les ids sans ligne : un dossier écrit avant cette règle, auquel il manque la ligne d'un id refusé, la
reçoit par un simple `--resume` avec le même modèle et les mêmes réglages.

`judges/trial-gptoss-off` (essai de 50 prompts, gpt-oss sans réflexion : sur un prompt, `"task": "design"`, hors
grille, à l'essai comme à la relance) date d'avant cette règle : l'utilisateur l'a retiré du dépôt, il reste comme
historique (judges/MODELS.md, judges/TRIAL.md) et n'est pas reconstruit.

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
standard (`ok (N abstentions)` s'il y en a), code de sortie 0. Dossier refusé : une faute par ligne sur la sortie
d'erreur, de la forme `labels.jsonl:LIGNE: ...`, `log.jsonl:LIGNE: ...` ou `meta.json: ...`, code de sortie 1.
