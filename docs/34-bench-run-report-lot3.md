# 34 — Lot 3 du banc : juges sur le jeu apparié, `run`, `report` (2026-10-30, validé)

Document de décisions, **rien n'est codé**. Validé par l'utilisateur le 2026-10-30 avec les avis donnés en § 5. Suite de
docs/32 (le kit en trois commandes, § 3.3 et § 3.4, les juges § 4, ce que le rapport doit dire § 5) et de docs/33 (lot
2, fusionné le 2026-10-30 : `data/prompts/base-v1.jsonl`, 360 lignes, 90 prompts de base × quatre langues, 22
corrections, aucun veto). Ce document fixe ce que le lot 3 fait tourner, ce qu'il calcule, ce qu'il publie, et ce qu'il
laisse aux lots suivants. Il ne change aucune décision de docs/32 ; là où il précise, il le dit.
Amendé le 2026-10-30 après-midi (§ 7, § 7.3 corrigé le soir) : l'heuristique de routage de LiteLLM entre comme
candidat, extraite et versionnée, dans une PR intermédiaire ; décisions 4 et 10 renvoient à § 7.
Amendé le 2026-10-30 au soir (§ 8) : la première PR est ouverte, la règle d'abstention des juges est confirmée.
Puis : première PR fusionnée, PR `lot-3c` ouverte (scombat/s1d1-bench#6), § 7.1 corrigé sur les mots-clés CJK.
Puis : Laya passe au nom à nom dans la PR B (§ 8, tranché le soir même).

## 0. Décisions proposées

1. **Les juges tournent sur `data/prompts/base-v1.jsonl` (360 lignes), tous, à neuf.** Les dossiers
   `judges/check-qwen3-off-<lang>` du lot 2 ne sont pas réunis pour en faire un juge : ils ont jugé les traductions
   **avant** les 22 corrections (c'est leur rôle, et leur fichier de prompts est `base-v1-<lang>.jsonl`, pas le jeu
   final). Ils restent ce qu'ils sont, des juges du banc sur un sous-ensemble, utilisables plus tard si `run` apprend
   les sous-ensembles ; le lot 3 n'en a pas besoin. Coût de la relance de Qwen3 : une heure. Raison : un juge du rapport
   doit avoir vu exactement le texte publié.
2. **Cinq dossiers de juges, trois dans la médiane principale.** La médiane principale est celle des trois modèles à
   réflexion plafonnée à 512 : `gemma4-cap512` (le réglage du juge de l'entraînement, docs/05), `qwen3-cap512`,
   `gptoss-cap512`. Les deux dossiers `gemma4-off` et `qwen3-off` sont la mesure de l'effet « réflexion » de docs/32 §
   4.3 sur les 360 prompts (l'essai l'a faite sur 50 : même palier à 84 % et 90 %) ; ils figurent dans le tableau des
   juges, pas dans la médiane principale, parce que deux réglages d'un même modèle ne sont pas deux juges indépendants.
   gpt-oss reste le troisième juge malgré son accord plus faible à l'essai (52 à 56 % sur le palier) : docs/32 demande
   trois juges locaux au minimum, et c'est le troisième modèle disponible qui lit le JSON sur 50 prompts sur 50 en
   `cap512`. Le rapport publie son accord un-contre-le-reste comme pour les autres ; si c'est lui qui s'écarte, le
   rapport le dit. Durées (moyennes de l'essai × 360) : Gemma cap512 3,7 h, Qwen cap512 2,3 h, gpt-oss 0,2 h, Gemma off
   1,2 h, Qwen off 1,0 h, soit **≈ 8,4 h, une nuit**, un modèle à la fois. `--max-tokens 1024`, température 0,
   `v7-single`, révisions de `judges/MODELS.md`.
3. **Deux vérités par prompt de base, comme docs/32 § 3.3** : la médiane **toutes langues** (12 votes : 3 juges × 4
   versions, puisque la règle 5 de la rubrique dit que les quatre versions ont les mêmes labels) est la vérité
   principale ; la médiane **par langue** (3 votes) sert à mesurer l'écart entre langues et les désaccords de langue.
   Médiane sur l'échelle ordinale trivial < low < medium < high < expert ; `null` compté à part (un prompt est «
   indécidable » si la majorité des votes est `null`, et sort des accords de palier) ; vote majoritaire pour `task` et
   `needs_tools`, égalité tranchée par l'ordre des juges écrit dans `run` et publié. Et toujours les deux chiffres de la
   décision 9 : médiane de tous, médiane sans Gemma 4 (deux juges : la médiane est la moyenne arrondie vers le bas des
   deux paliers, et le rapport le dit).
4. **Les classifieurs du lot 3 : s1d1 depuis le Hub, Laya, les témoins.** s1d1 en quatre variantes publiées, poids du
   Hub à révision épinglée, par le `s1d1_router.py` de la carte (le fichier est dans le dépôt Hub) : `full` fp16
   (PyTorch, le modèle de référence) et `full`, `compact`, `small`, `tiny` en ONNX int8 (CPU, `reduce_range`, révisions
   corrigées du 2026-10-25). Laya english et multilingual avec la question `tier` et les critères copiés de LiteLLM
   comme docs/27 (adaptateur `bench/laya_adapter.py` du dépôt de recherche, qui déménage), dans un groupe de dépendances
   à part. Témoins : `random` (10 tirages à graine fixe, moyenne et bande), `length` (palier par déciles de la longueur
   en caractères du prompt, sur le jeu lui-même ; score = rang normalisé), `constant-low`. Chaque classifieur écrit
   `classifiers/<nom>/labels.jsonl` + `meta.json` au format d'un juge, plus `score` et `needs_tools_probability` quand
   il les a (`provider: "system"`, nouvelle valeur). Clef-flash et le classifieur NVIDIA restent hors du lot 3 (docs/32
   § 8 : non vérifiés) ; les trois graines de `small` ne sont pas sur le Hub, la bande σ de docs/18 est reprise comme
   nombre (σ ≈ 0,005 QWK), pas remesurée.
   **Amendement 2026-10-30** : un sixième candidat, `litellm-heuristic`, le scorer par défaut de l'Auto Router de
   LiteLLM, extrait et versionné (§ 7).
5. **Les juges sont aussi des classifieurs, un-contre-le-reste** (docs/32 décision 10) : chaque juge noté contre la
   médiane des autres juges de la médiane principale. C'est le plafond.
6. **`run` produit `results/base-v1.parquet`, dérivé et recalculable** : une ligne par id, une colonne par juge et par
   classifieur et par sortie, plus les médianes et l'écart. À la différence des dossiers `judges/` et `classifiers/`
   (immuables, seulement ajoutés), ce fichier et le rapport se régénèrent à chaque `run` ; ce qui est rejouable, ce sont
   les entrées. `run` ne charge que les dossiers dont `prompts_sha256` est celui du jeu demandé, et écrit la liste des
   dossiers chargés dans `results/base-v1.run.json` (noms, dates, révisions).
7. **`report` écrit `report/base-v1/index.md`, les figures en SVG et chaque figure avec son CSV**, matplotlib, les
   analyses de docs/32 § 3.4 au complet, toutes avec IC à 95 % par rééchantillonnage des prompts de base (1 000 tirages,
   le tirage porte sur les 90 `base_key`, les quatre versions suivent, pour ne pas compter quatre fois le même contenu).
   Le chiffre principal est le QWK du palier contre la médiane toutes langues des trois juges `cap512` ; les mises en
   garde de docs/32 § 2 sont recopiées depuis un gabarit, et la phrase de docs/32 § 5 est écrite telle que les chiffres
   la donnent, négative si c'est ce qui sort. **Aucune phrase « meilleur que »** sans que l'IC de l'écart exclue zéro et
   dépasse 2 σ.
8. **Le score de s1d1 est noté contre l'ancre du palier médian** (trivial 0,01, low 0,25, medium 0,50, high 0,75, expert
   1,00, docs/32 § 1) : MAE, Spearman, diagramme de fiabilité sur dix cases de score, erreur de calibration attendue. Le
   palier prédit de s1d1 est l'ancre la plus proche du score, comme la carte du Hub.
9. **L'écart entre langues se mesure à contenu identique** : pour chaque `base_key`, le palier prédit par s1d1 sur ses
   quatre versions, et la médiane par langue ; on publie le QWK par langue contre la médiane toutes langues, et le taux
   de prompts de base où s1d1 change de palier d'une langue à l'autre. Les juges ont le même tableau : c'est le «
   désaccord de langue » de docs/32 § 3.3, qui dit si l'écart vient de la grille, de la traduction ou du juge.
10. **Deux PR pour le lot** : la première livre `bench run`, les adaptateurs de classifieurs, les cinq dossiers de juges
    et `classifiers/`, `results/base-v1.parquet` ; la seconde livre `bench report` et `report/base-v1/`. Raison : la
    première est longue en machine et courte en code, la seconde l'inverse ; une PR de 8 h de séries et de figures
    serait illisible.
    **Amendement 2026-10-30** : une PR intermédiaire `lot-3c` (l'heuristique LiteLLM, § 7) part de `main` après la
    fusion de la première et précède la seconde, pour que le rapport la couvre dès sa première lecture.
11. **Le dépôt reste privé jusqu'à la fusion de la seconde PR** (docs/32 décision 12). L'ouverture est une décision à
    part de l'utilisateur, qui demande avant : la vérification de la couverture Apache-2.0 des prompts Arena-Hard
    (docs/32 § 8), l'épinglage de `mlx-lm` contre Dependabot (un bump change l'identité des juges), et une relecture du
    rapport.

## 1. Ce qui a été vérifié (2026-10-30)

- `data/prompts/base-v1.jsonl` : 360 lignes, 90 par langue, `meta.parquet` avec `corrected` vrai sur 22 ids (fra 9, rus
  6, zho 7), `translator` nul en anglais ; `.sha256` en place ; 9 lots de 40. Licences par ligne : apache-2.0 248,
  cc-by-sa-3.0 72, CC-BY-4.0 40 (les dix prompts écrits à la main × 4).
- Les dossiers `judges/check-qwen3-off-<lang>` ont pour `prompts_sha256` celui de `base-v1-<lang>.jsonl`, les
  traductions **non corrigées** ; d'où la décision 1.
- Durées de l'essai (`judges/TRIAL.md`, 50 prompts, M4 Max 128 Go, `mlx-lm` 0.32.2) : moyennes par prompt Gemma off 11,6
  s, Gemma cap512 37,3 s, Qwen off 9,5 s, Qwen cap512 22,5 s, gpt-oss cap512 1,9 s. Le jeu `base-v1` a 28 Arena-Hard sur
  90 (31 %, l'essai en avait 32 %) et dix prompts experts longs : la projection par 360 est du même ordre.
- Accords de l'essai (50 prompts, palier exact) : Gemma/Qwen 88 % (off) et 74 % (cap512) ; gpt-oss contre les deux 52 à
  56 % ; Gemma off contre cap512 84 %, Qwen 90 %. Aucun `expert` et 0 ou 1 `needs_tools` à l'essai : le jeu `base-v1`,
  avec dix prompts experts et 17 `needs_tools` au pré-tri, remplit mieux le haut de la grille.
- Le code s1d1 : `release/s1d1-router/s1d1_router.py` (Hub, Apache-2.0) rend `complexity_score`, `complexity_tier`
  (ancre la plus proche), `task`, `needs_tools_probability`, `needs_tools`, avec trois backends (torch, MLX, ONNX
  Runtime) ; `bench/s1d1_adapter.py` du dépôt de recherche le pilote par lots triés par longueur avec cache. Les
  variantes du Hub : `scombat/s1d1-router-6src-{full,compact,small,tiny}` et leurs `-onnx-int8` (révisions corrigées
  `small` 33c7f34, `full` 068d0d2, `compact` 7e87870 ; `tiny` à relire sur la carte).
- Laya : adaptateur de docs/27 (`bench/laya_adapter.py`, Laya 0.4.15 dans un venv à part, question et critères de
  LiteLLM lus le 2026-10-25). Non revérifié depuis.
- Le format des juges (`judge_file.py`) : `provider` ∈ local, chat, api, human ; `instructions` ∈ v7-single, v7-batch ;
  `read_labels` rend id → ligne ; `bench agree` fait déjà l'accord deux à deux (palier exact, ± 1, tâche, outils).

## 2. Les séries de juges

Ordre, un modèle chargé à la fois, en arrière-plan, journal dans le scratchpad, reprise par `--resume` : `qwen3-off` (1
h, le premier parce qu'il est court et sert de contrôle), `gptoss-cap512` (0,2 h), `qwen3-cap512` (2,3 h), `gemma4-off`
(1,2 h), `gemma4-cap512` (3,7 h). Dossiers `judges/base-v1-<famille>-<réglage>`, `--author scombat`, `notes` qui dit le
rôle (médiane principale ou effet réflexion). Chaque dossier passe `bench validate` avant le commit ; un dossier
incomplet (réponse illisible, hors énumération) n'est pas complété à la main : il est signalé, et l'utilisateur décide
comme pour `trial-gptoss-off`. La leçon du lot 2 s'applique : le contrôleur attend la fin de chaque chaîne de son côté
(PID), et relance la session si elle s'endort.

Ce que l'on regarde avant de continuer (contrôle après `qwen3-off`) : distribution des paliers (les dix experts doivent
sortir `high` ou `expert`), `needs_tools` vrai sur une quinzaine de prompts, aucune réponse illisible.

## 3. Les classifieurs et les témoins

`bench classify <adaptateur> --prompts data/prompts/base-v1.jsonl --out classifiers/<nom> [options]`, même discipline
que `bench judge` (fichiers jamais réécrits, meta avant la première prédiction, révision épinglée, SHA-256 du jeu).

- `s1d1` : `--model <dépôt Hub ou dossier> --revision <sha> --backend auto|onnx|torch|mlx`, lots triés par longueur,
  troncature à 1 024 jetons (celle du modèle). Cinq dossiers : `s1d1-full-fp16`, `s1d1-full-int8`, `s1d1-compact-int8`,
  `s1d1-small-int8`, `s1d1-tiny-int8`. Dépendances dans un groupe `s1d1` : `onnxruntime`, `tokenizers`, `numpy`, et
  `torch` pour fp16 seulement.
- `laya` : groupe `laya`, à part ; si Laya 0.4.15 ne s'installe pas proprement dans l'environnement du banc, les deux
  dossiers sont produits par l'adaptateur de docs/27 et versés comme classifieurs `replayable: true` avec le venv décrit
  dans `notes`, plutôt que de tordre `pyproject`.
- Témoins : `random` (graine 0, dix tirages ; `labels.jsonl` porte le premier, le rapport calcule la bande sur les dix),
  `length` (déciles sur le jeu, correspondance déciles → paliers 2/3/3/1/1 pour suivre la proportion attendue
  trivial/low/medium/high/expert du pré-tri, écrite dans `meta`), `constant-low`. Les témoins sont le plancher ; s'ils
  ne sont pas nettement sous s1d1, le rapport le dit en premier.

## 4. `run` et `report`

- `bench run --prompts data/prompts/base-v1.jsonl --judges judges/base-v1-* --main
  gemma4-cap512,qwen3-cap512,gptoss-cap512 --classifiers classifiers/* --out results/base-v1.parquet` : refuse un
  dossier dont le SHA-256 n'est pas celui du jeu, refuse un id manquant, écrit `results/base-v1.run.json`.
- `bench report --results results/base-v1.parquet --meta data/prompts/base-v1.meta.parquet --out report/base-v1` :
  tableaux Markdown et CSV (accord de chaque classifieur et de chaque juge avec la médiane : tous, sans Gemma 4,
  un-contre-le-reste ; par langue, par source, par origine, par palier médian), figures SVG (barres QWK avec IC, plafond
  et plancher en lignes, bande σ ; matrices de confusion normalisées ; fiabilité du score ; écart entre juges par palier
  ; QWK par langue ; désaccords de langue), `index.md` avec les mises en garde du gabarit. Analyses : QWK, alpha de
  Krippendorff ordinal, exactitude à ± 1 ; F1, AUROC, MCC et précision-rappel pour `needs_tools` ; exactitude et F1
  macro pour `task` ; MAE, Spearman, fiabilité et ECE pour le score ; différence appariée entre classifieurs avec IC
  bootstrap. Implémentation sans dépendance nouvelle hors `matplotlib` (QWK, alpha, AUROC et bootstrap écrits dans le
  dépôt, avec leurs tests contre des valeurs connues) ; `scikit-learn` n'entre pas pour trois formules.
- Le rapport est daté et nomme ses entrées (dossiers, révisions) ; un nouveau juge ou classifieur = relancer `run` et
  `report`, pas de recalcul à la main.

## 5. Points à trancher par l'utilisateur

Tranchés le 2026-10-30 : l'utilisateur a retenu l'avis donné à chaque point (cap512, gpt-oss gardé pour ce rapport, Laya
dans le lot 3, deux PR, pas de contribution par chat avant le rapport). Les questions restent ci-dessous pour mémoire.

1. La médiane principale sur les trois `cap512` (décision 2), ou sur les deux `off` plus gpt-oss ? Mon avis : les trois
   `cap512`, parce que c'est le réglage du juge de l'entraînement et que les deux réglages sont mesurés de toute façon.
2. gpt-oss-20b reste-t-il troisième juge, vu son accord à l'essai ? Mon avis : oui pour ce rapport, faute d'un quatrième
   modèle mesuré ; si le rapport montre qu'il tire la médiane, le lot suivant essaie GLM 5.2 ou DeepSeek V4 flash
   (docs/32 § 4.1) et on refait `run`, sans retoucher les dossiers existants.
3. Laya dans le lot 3, ou dans un lot « concurrents » avec Clef-flash et NVIDIA ? Mon avis : dans le lot 3, parce que
   l'adaptateur existe et qu'un rapport sans aucun concurrent dit moins ; Clef-flash et NVIDIA plus tard.
4. Deux PR (décision 10) ou une seule ? Mon avis : deux.
5. Les contributions par chat (`v7-batch`, `example-chat`) sont-elles attendues avant le rapport ? Mon avis : non ; le
   rapport a une annexe vide prête à les recevoir, et la mesure « seul contre lot » de docs/32 § 4.3 attend la première
   contribution réelle.

## 6. Non vérifié

- La durée réelle sur 360 prompts dont dix experts longs et 28 Arena-Hard : la projection vient de 50 prompts.
- Que gpt-oss-20b en `cap512` lise les 360 (l'essai : 50 sur 50 ; en `off`, une tâche hors énumération).
- Que `s1d1_router.py` du Hub et ses backends s'installent dans l'environnement du banc sans conflit de versions
  (`onnxruntime`, `tokenizers`, `torch`) avec `mlx-lm` 0.32.2 ; et que l'ONNX int8 donne les mêmes paliers sur le Mac
  (ARM) que sur x86 (docs/28 a corrigé `reduce_range` pour x86).
- Que Laya 0.4.15 tourne encore avec les critères de LiteLLM relus ; LiteLLM peut avoir changé la question depuis le
  2026-10-25.
- La couverture Apache-2.0 des prompts Arena-Hard pour une redistribution publique (docs/32 § 8), à relire avant
  l'ouverture, pas pour le lot 3.

## 7. Amendement du 2026-10-30 (après-midi) : l'heuristique de LiteLLM comme candidat

Décidé en discussion le 2026-10-30 : le bon concurrent de s1d1, avec Laya, est **le routeur intégré de LiteLLM tel
qu'un opérateur l'obtient sans rien installer**, c'est-à-dire le scorer heuristique, réglage par défaut
(`classifier_type: heuristic`). Les autres `classifier_type` de LiteLLM (`llm` et sa rubrique intégrée, `jev`,
`capability`, `llm_v2`) et les routeurs ouverts publiés récemment viendront plus tard, chacun avec sa note ; la rubrique
`llm` et les cas d'évaluation que LiteLLM livre avec (`evals/eval_complexity_router.py`) sont remis à ce moment-là.

### 7.1 Ce qu'est l'heuristique (vérifié au tag `v1.106.6`, commit `61c6ab78`, le 2026-10-30)

Fichiers `litellm/router_strategy/complexity_router/complexity_router.py` (4 976 lignes) et `config.py` (2 375). Le
scorer est la méthode `ComplexityRouter._score_and_classify(prompt, system_prompt)` et ses aides `_estimate_tokens`
(longueur en caractères divisée par 4), `_score_token_count`, `_keyword_matches`, `_score_keyword_match`,
`_score_multi_step`, `_score_question_complexity`. Sept dimensions pondérées (`DEFAULT_DIMENSION_WEIGHTS` : code 0,30,
marqueurs de raisonnement 0,25, termes techniques 0,25, jetons 0,10, indicateurs de simplicité 0,05, multi-étapes 0,03,
points d'interrogation 0,02), listes de mots-clés de `config.py` (`DEFAULT_CODE_KEYWORDS`, `DEFAULT_REASONING_KEYWORDS`,
`DEFAULT_TECHNICAL_KEYWORDS`, `DEFAULT_SIMPLE_KEYWORDS`, toutes en écriture latine, aucune entrée CJK : corrigé le
2026-10-30 au soir après l'extraction, la règle « CJK en sous-chaîne » du code n'a donc aucun mot-clé à servir), seuils
de jetons 15 et 400, bornes de score `simple_medium` 0,15, `medium_complex` 0,35, `complex_reasoning` 0,60, et une
règle de priorité : deux marqueurs de raisonnement ou plus, au-dessus d'un plancher de score, donnent `REASONING` quel
que soit le score. Quatre paliers : `SIMPLE`, `MEDIUM`, `COMPLEX`, `REASONING`. Aucun appel de modèle, moins d'une
milliseconde. Seule la demande de l'utilisateur est notée, jamais le prompt système : sur le banc, `prompt` = la ligne
de `base-v1.jsonl`, `system_prompt = None`, ce qu'un opérateur obtient pour une requête à un seul tour.

### 7.2 Exactement leur code, extrait et versionné (décision de l'utilisateur)

- La classe `ComplexityRouter` dépend de tout le paquet `litellm` (cache, journalisation, httpx, pydantic…). Le banc
  n'installe pas `litellm` : il **extrait mot pour mot** les méthodes du scorer et les constantes de `config.py` dans
  `src/s1d1_bench/vendor/litellm_heuristic/`, avec le fichier de licence de LiteLLM au même tag et un `NOTICE` qui donne
  le tag, le commit, les chemins et les plages de lignes copiées. Aucune réécriture, aucune « amélioration » ; les seuls
  changements sont ceux qui rendent le code importable sans `litellm` (les imports), listés dans le `NOTICE`.
- **Preuve d'identité** : à l'extraction, une seule fois, le vrai paquet `litellm==1.106.6` tourne dans un venv jetable
  sur les 360 lignes ; sa sortie (palier, score brut, signaux) est versée dans
  `tests/fixtures/litellm-heuristic-v1.106.6-base-v1.jsonl`, et un test vérifie que la copie extraite la reproduit à
  l'identique. Le paquet n'entre jamais dans `pyproject`. Changer de version de LiteLLM = nouvelle extraction, nouveau
  `NOTICE`, nouvelle fixture, nouveau dossier de classifieur ; l'ancien reste.
- Dossier `classifiers/litellm-heuristic/`, format de la décision 4 : `model` = `litellm complexity_router heuristic`,
  `revision` = le commit du tag, `provider: "system"`, `instructions: "none"`, `thinking: "none"`, `task` et
  `needs_tools` nuls. Le palier et le score bruts de LiteLLM (`SIMPLE`, 0,2375…) et les signaux vont dans `log.jsonl`,
  ligne par ligne ; `labels.jsonl` porte le palier et le score **traduits** (§ 7.3).

### 7.3 Les quatre paliers et le score brut : le moins de manipulation possible

Corrigé le 2026-10-30 au soir : une première version de ce paragraphe reportait le score brut de LiteLLM sur des
plages de l'échelle v7 ; ce mapping venait d'une prémisse fausse de ma part (« LiteLLM ne rend que quatre paliers, pas
de score »), et l'utilisateur a rappelé qu'un banc doit manipuler les chiffres le moins possible. Règle retenue :

- **Paliers** : correspondance de nom à nom, sans arithmétique. `SIMPLE` → low, `MEDIUM` → medium, `COMPLEX` → high,
  `REASONING` → expert. LiteLLM n'a pas de trivial : sur les prompts trivial de la vérité il est au mieux à un cran, et
  le rapport le dit, comme il dit que s1d1 ne prédit jamais expert.
- **Score** : le score brut de LiteLLM, tel quel, sans report sur l'échelle v7 et sans borne (il va de −0,15 à
  0,925). Comme la règle du kit impose un `score` dans [0, 1] et sur l'échelle v7, il n'est **pas** écrit dans
  `labels.jsonl` : il va avec le palier brut, les signaux et la cause dans `decisions.jsonl`, à côté, et `bench run`
  en fait une colonne `raw_score`. Il ne sert que dans les mesures qui ne regardent que l'ordre (corrélation de rang,
  AUROC) ; pas de calibration ni d'erreur absolue pour ce candidat, puisque son score n'est pas sur l'échelle de la
  vérité. `task` et `needs_tools` nuls. (Précision du soir : « `log.jsonl` » en § 7.2 se lit « `decisions.jsonl` ».)

### 7.4 Ce que le rapport en dit

`litellm-heuristic` est un candidat au même titre que Laya et les variantes de s1d1 : QWK, exact, ±1 sur les deux
vérités, corrélation de rang de son score brut, par langue, avec les mêmes intervalles (bootstrap sur les 90 clés).
Deux lectures attendues,
écrites avant de voir les chiffres : ses listes de mots-clés sont anglaises, donc son score par langue devrait tomber
en français, en russe et en chinois ; et il devrait se tenir entre le témoin `length` et s1d1. Si l'une des deux ne se
vérifie pas, le rapport le dit en premier. `task` et `needs_tools` : non mesurés pour ce candidat.

### 7.5 Non vérifié

- Que l'extraction tienne en un module court : `_score_and_classify` lit la configuration par `self.config`
  (`token_thresholds`, `tier_boundaries`, `dimension_weights`, listes de mots-clés, dimensions personnalisées vides par
  défaut) ; la copie fixe les valeurs par défaut de `config.py` et rien d'autre.
- Le plancher de la règle de priorité (`_effective_reasoning_override_min_score`) et la valeur exacte de
  `DEFAULT_TOKEN_THRESHOLDS`, à lire dans le code à l'extraction, pas de mémoire.
- La licence du cœur de LiteLLM au tag (le dossier `enterprise/` est sous une autre licence) : le fichier `LICENSE` du
  tag est copié tel quel, et la question reste celle de l'utilisateur avant l'ouverture du dépôt.

## 8. Amendement du 2026-10-30 (soir) : première PR ouverte, règle d'abstention confirmée

La première PR du lot (`lot-3a`, scombat/s1d1-bench#5, 17 commits) est ouverte : cinq dossiers de juges sur les 360
lignes, dix dossiers de classifieurs, `results/base-v1.parquet` par `bench run`. Durées réelles, un modèle à la fois :
Qwen off 57 min, gpt-oss cap512 12 min, Qwen cap512 3 h 02, Gemma off 1 h 18, Gemma cap512 4 h 29, soit 9 h 57 contre
les 8,4 h projetées depuis l'essai (les dix prompts expert et les 28 Arena-Hard allongent la réflexion plafonnée).

**Règle d'abstention, confirmée par l'utilisateur le 2026-10-30 au soir.** Sur le prompt chinois de synthèse de HCl
(`3ce175ed3ef6b933-zho`), gpt-oss refuse de répondre, deux fois en décodage glouton. Le dossier `base-v1-gptoss-cap512`
compte donc 359 réponses et une ligne d'abstention : `complexity`, `task` et `needs_tools` à `null`, plus
`"abstained": "json_error"`, écrite par `bench judge --resume` à partir du journal. `validate` l'annonce (`ok (1
abstention)`), `agree` sort l'id de la paire et compte l'abstention, `run` la liste dans `run.json` ; la vérité se
calcule sur les votes restants. Seuls un refus (`json_error`) et une réponse coupée (`truncated`) sont des abstentions :
une exception du moteur (`error`) ne laisse pas de ligne, `--resume` repose la question et `validate` signale l'id
manquant, pour qu'une panne passagère ne devienne pas un enregistrement permanent. La décision 2 ne change pas : gpt-oss
reste le troisième juge, le rapport dit son abstention avec son accord.

**Autres écarts au plan, acceptés** : médiane d'un nombre pair de votes = moyenne des deux rangs centraux arrondie vers
le bas ; égalité de `nearest_anchor` tranchée vers le palier bas ; palier de s1d1 pris à `decode()` (plafonné à high) ;
séries int8 small, compact et full épinglées aux révisions de la carte, tiny et full fp16 à la tête du dépôt avec la
révision dans `meta` ; `--main` prend les noms de dossier exacts.

**Laya passe au nom à nom (tranché par l'utilisateur le 2026-10-30 au soir).** Laya répond à la question à quatre
paliers de LiteLLM ; la PR A le convertissait par `nearest_anchor(attendu / 3)` (SIMPLE → trivial, MEDIUM → low,
COMPLEX → high), alors que § 7.3 associe les quatre noms un à un pour l'heuristique. Deux candidats qui répondent à la
même question se lisent avec la même table : dans la PR B, Laya prend le palier le plus probable des quatre et le
traduit nom à nom (SIMPLE → low, MEDIUM → medium, COMPLEX → high, REASONING → expert, jamais trivial) ; ses quatre
probabilités sont gardées telles quelles dans un `decisions.jsonl`, comme le score brut de LiteLLM, pour les métriques
de rang ; `labels.jsonl` n'a plus de `score`. Les deux dossiers `laya-*` de la PR A ne sont pas modifiés : la PR B en
ajoute deux nouveaux (≈ 22 s) et le tableau ne retient que ceux-là, en disant pourquoi.

**Hors code** : les trois jobs GitHub Actions de la PR ne démarrent pas (facturation du compte : « recent account
payments have failed or your spending limit needs to be increased ») ; la CI a été rejouée en local (ruff, 584 tests,
`validate` sur 29 dossiers, gitleaks) avant l'ouverture.
