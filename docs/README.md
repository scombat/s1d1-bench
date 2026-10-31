# Les décisions du projet

Les trois documents de ce dossier, `32-open-bench-kit.md`, `33-translation-lot2.md` et `34-bench-run-report-lot3.md`,
sont des copies, à l'identique, de trois pièces du cahier de décisions du projet s1d1 (un dépôt de recherche privé).
La numérotation du cahier est conservée, d'où des numéros qui sautent. Ils ont été copiés le 2026-10-31 et sont
sous licence [CC-BY-4.0](../data/LICENSE). Le banc les cite sous la forme `docs/NN`.

## Ce que quatre documents non copiés apportent au banc

- `docs/05` : le banc des juges sur HF Jobs. Gemma 4 avec une réflexion plafonnée à 512 est retenu comme juge
  d'entraînement. Le plafond de 512 a été testé contre 1 024, et 512 a été gardé.
- `docs/17` : le banc par langue du routeur sur son jeu de test, sur 78 langues. Le QWK par langue suit
  l'étalement des labels de la langue.
- `docs/18` : trois graines de la variante small. L'écart-type est d'environ 0,005 de QWK (0,009 sur le F1 de
  `needs_tools`) ; c'est le σ de la règle des 2 σ du rapport.
- `docs/27` : le banc de comparaison avec les autres routeurs. Il porte le diagnostic de RouteLLM et la lecture de la
  question à quatre paliers de LiteLLM posée à Laya.

## Les numéros qui restent privés

Les numéros `docs/01`, `docs/06`, `docs/07`, `docs/25` et `docs/28`, cités à l'intérieur des trois copies, restent
privés. Ils ne sont pas nécessaires pour lire le banc.
