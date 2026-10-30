# Les fichiers de juges

Un juge = un dossier `judges/<nom>/` de deux fichiers. `judges/example-chat/` en est un exemple complet, sur trois
prompts fictifs (`data/prompts/example.jsonl`).

## `meta.json`

Un objet JSON avec ces 13 clés, toutes obligatoires :

| clé | valeur |
|---|---|
| `judge` | nom libre du juge |
| `model` | identifiant exact du modèle |
| `provider` | `local`, `chat`, `api`, `human` ou `system` (un classifieur, voir plus bas) |
| `revision` | commit ou version si connue, sinon `"unknown"` |
| `date` | date du jugement, `AAAA-MM-JJ` |
| `temperature` | si connue, sinon `"unknown"` (toute valeur acceptée) |
| `thinking` | si connu, sinon `"unknown"` (toute valeur acceptée) |
| `instructions` | `v7-single` ou `v7-batch` (la consigne de `consigne/`), ou `none` pour `system` et pour lui seul |
| `prompts_sha256` | empreinte du fichier de prompts jugé : le contenu de `data/prompts/<jeu>.sha256` |
| `author` | qui verse le fichier |
| `replayable` | booléen JSON : pour `local`, `true` seulement à température 0 et révision figée ; les classifieurs du banc (`system`) sont `true` |
| `licence` | exactement `CC-BY-4.0` : un fichier sous une autre licence ne peut pas être fusionné |
| `notes` | texte libre (pour une correction humaine : ce qui a été corrigé) |

Seules trois clés de plus sont tolérées, écrites par `bench judge` : `quantization`, `mlx_lm_version` et `max_tokens`
(un entier strictement positif). Toute autre clé est refusée.

## `labels.jsonl`

Une ligne par prompt du fichier de prompts désigné par `prompts_sha256`, un objet JSON avec quatre champs (et, pour un
classifieur ou un juge qui les rend, deux champs facultatifs : `score` et `needs_tools_probability`) :

```json
{"id": "bad9ea434b83cd45-eng", "complexity": "trivial", "task": "information_seeking", "needs_tools": false}
```

- `id` : une chaîne, l'`id` d'une ligne du fichier de prompts ; tous présents, aucun en trop, aucun en double.
- `complexity` : `trivial`, `low`, `medium`, `high`, `expert`, ou `null` (le JSON `null`, pas la chaîne) si le prompt ne
  peut pas être jugé.
- `task` : l'une des 19 tâches de la grille v7 (`information_seeking`, `reasoning`, ..., `chat`, `other`) ; `null` n'est
  accepté que pour `provider: system`.
- `needs_tools` : `true` ou `false`, un vrai booléen JSON (ni `"true"`, ni `1`) ; `null` n'est accepté que pour
  `provider: system`.
- `score` et `needs_tools_probability` (facultatifs, dans n'importe quel dossier) : absents, ou un nombre de 0 à 1 (ni
  booléen, ni chaîne).

Aucun contrôle de qualité du contenu : un juge médiocre se voit dans son accord avec les autres.

### Les abstentions

Quand le modèle ne rend pas d'étiquette lisible (état final `json_error` ou `truncated` dans `log.jsonl`, par exemple
un refus ; la relance n'a lieu que pour une réponse illisible qui n'a pas été coupée par `--max-tokens` : `json_error` est
l'état après cette relance, `truncated` celui d'une réponse coupée, sans relance), `bench judge` écrit une ligne d'abstention plutôt que rien :

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

## Les dossiers de classifieurs

`classifiers/<nom>/` est un dossier de juge pour un classifieur (`bench classify`) : les mêmes deux fichiers, `meta.json` et
`labels.jsonl`, validés par le même `bench validate`. Différences :

- `provider` est `system`, et `instructions` est `none` ; `bench validate` exige les deux ensemble (ni l'un sans l'autre) ;
  `bench classify` écrit en plus `thinking: none` ;
- `score` et `needs_tools_probability` sont facultatifs, absents ou dans [0, 1] ;
- `task` et `needs_tools` peuvent être `null` pour ce que le classifieur ne prédit pas (Laya et les témoins `length` et
  `constant` ne prédisent ni l'un ni l'autre) ;
- une ligne `abstained` est refusée : un classifieur ne s'abstient jamais ;
- pas de `log.jsonl` ;
- `decisions.jsonl` est un fichier facultatif du dossier : les décisions brutes d'un adaptateur, une ligne par id, avec
  `id`, `tier`, `score`, `signals` et `cause`. `bench validate` l'ignore ; `bench run` le lit (colonnes
  `<nom>.raw_score` et `<nom>.raw_tier`) et le refuse s'il manque un id, s'il en a un inconnu ou un doublon ;
- `score` est absent de `labels.jsonl` quand le score propre du classifieur n'est pas sur l'échelle v7 : c'est le cas de
  `litellm-heuristic`, dont le score (qui peut être négatif) n'est que dans `decisions.jsonl`, pour les mesures de rang.

## Un fichier accepté ne change plus

Un fichier accepté ne change plus. Une correction est un nouveau dossier `judges/<nom>-corrige-<auteur>/`, avec
`provider: human` et des `notes` qui disent ce qui a été corrigé. Les deux apparaissent dans le rapport.

Les dossiers `check-*` sont des juges du banc sur un sous-ensemble du jeu (les 90 prompts de base, dans une langue) ;
les dossiers `witness-*` (traductions témoins) et `presort-*` (pré-tri) n'en sont pas.

## Les dossiers `base-v1-*`

Les cinq juges du rapport de `base-v1`, tous sur `data/prompts/base-v1.jsonl` (360 prompts, quatre langues) avec la consigne
`v7-single`, `--max-tokens 1024` et la température 0. Leurs `notes` disent le rôle ; la lecture d'ensemble (durées, répartition des
paliers, accords deux à deux, commandes pour les refaire) est dans la section « Les juges du rapport et les classifieurs » du
[README](../README.md).

| dossier | modèle | réflexion | rôle |
|---|---|---|---|
| `base-v1-gemma4-cap512` | `mlx-community/gemma-4-31b-it-8bit` | `cap:512` | médiane principale |
| `base-v1-qwen3-cap512` | `mlx-community/Qwen3.8-27B-8bit` | `cap:512` | médiane principale |
| `base-v1-gptoss-cap512` | `mlx-community/gpt-oss-20b-MXFP4-Q8` | `cap:512` | médiane principale ; une abstention |
| `base-v1-gemma4-off` | `mlx-community/gemma-4-31b-it-8bit` | `off` | paire « effet de la réflexion » avec `gemma4-cap512`, hors médiane principale |
| `base-v1-qwen3-off` | `mlx-community/Qwen3.8-27B-8bit` | `off` | paire « effet de la réflexion » avec `qwen3-cap512`, hors médiane principale |

`base-v1-gptoss-cap512` est le dossier du schéma d'abstention : 360 lignes, dont 359 étiquettes et une ligne `"abstained": "json_error"` pour l'id
`3ce175ed3ef6b933-zho` (un refus sur le prompt chinois de synthèse de l'acide chlorhydrique) ; `bench validate` affiche
`ok (1 abstention)`. Les quatre autres dossiers ont 360 étiquettes, sans abstention. Gemma 4 `cap512` a un palier `null` pour un id : c'est la
sortie du juge, pas une abstention.

## Vérifier un dossier

```
uv run bench validate judges/<nom> [--prompts data/prompts]
```

`--prompts` désigne le dossier des fichiers de prompts (`data/prompts` par défaut). Dossier accepté : `ok` sur la sortie
standard (`ok (N abstentions)` s'il y en a), code de sortie 0. Dossier refusé : une faute par ligne sur la sortie
d'erreur, de la forme `labels.jsonl:LIGNE: ...`, `log.jsonl:LIGNE: ...` ou `meta.json: ...`, code de sortie 1.
