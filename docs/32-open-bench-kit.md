# 32 — Banc de fidélité de la note : ouvert, rejouable, par fichiers de juges (2026-10-28, validé)

Document de décisions, **rien n'est codé, aucun dépôt n'est créé**. Suite de docs/27 (banc RouteLLM : s1d1 indiscernable
d'un routeur à la longueur du prompt, banc maison différé), docs/17 (banc par langue), docs/18 (trois graines), docs/25
(boucle capturer → étiqueter → comparer). Déclencheur : l'Auto Router de Cloudflare (InfoQ, 2026-10-26) publie un banc
interne de 97 tâches, sans témoin, sans concurrent, dont les deux modèles de référence ne sont pas dans le pool par défaut.
Demande de l'utilisateur, 2026-10-28 : un banc **honnête mais ouvert**, **rejouable et productible**, qui mesure **la note**
de s1d1 (palier, score, tâche, `needs_tools`) et **pas le choix d'un modèle**, celui-ci restant l'affaire de LiteLLM ou
du déployeur ; un kit que n'importe qui peut rejouer avec son propre juge ; un dépôt à part.

Validé par l'utilisateur le 2026-10-28 (§ 0), après ses réponses aux neuf questions du § 7.

**Amendement du 2026-10-29** (§ 3.1, étape 3) : le choix à la main des prompts de base est remplacé par un **tirage
aléatoire stratifié à graine fixe** sur les étiquettes du pré-tri, avec un veto borné à deux règles, préparé par
l'assistant et confirmé par l'utilisateur. Raison : ni l'utilisateur ni l'assistant ne doivent décider ce qu'est un
`high` (c'est ce que s1d1 a appris de Gemma 4) ; la vérité du banc est la médiane des juges, calculée après, donc une
erreur de strate déséquilibre la taille d'une case sans fausser la mesure. Le lot 1 est fusionné (PR #1 de
`s1d1-bench`, 2026-10-29) ; l'en-tête « rien n'est codé » ci-dessus date du 2026-10-28.

## 0. Décisions (validées par l'utilisateur le 2026-10-28)

1. **Le banc mesure la fidélité de la note à la grille de docs/06, pas le routage.** Raison : c'est ce que s1d1 prédit ;
   docs/27 a montré qu'un banc « quel modèle réussit » sur des questions académiques ne mesure pas cela.
2. **La vérité de référence est la médiane de plusieurs juges indépendants**, et l'accord entre juges est le plafond.
   Raison : la difficulté est une convention (la grille) ; la seule vérité disponible est le jugement de ceux qui l'appliquent.
3. **Un document de juge est un fichier au format unique, produit par une commande ou par un chat**, avec un en-tête qui
   dit qui a jugé, quand, avec quelle version et quelle consigne. Raison : rejouable (on refait le fichier) et ouvert (on
   verse le sien).
4. **Un fichier de juge publié ne se modifie jamais : on en ajoute un.** Une correction humaine d'un juge est un nouveau
   fichier, avec son auteur ; l'original reste. Raison : sans cela, aucun chiffre publié n'est rejouable.
5. **La ligne de base rejouable est un jeu de juges à poids ouverts tournant en local** (température 0, révision figée,
   consigne figée), dont le juge de l'entraînement (Gemma 4 31B-it), **jamais seul**. ChatGPT, Claude, un humain, sont
   des contributions au même format, marquées « non rejouables ». Raison : les modèles par API changent sans graine.
6. **Trois commandes** : `export` (prompts + consigne), `run` (fusion des juges, puis chaque classifieur en colonne),
   `report` (tableaux, graphiques, données brutes). Un juge ou un classifieur de plus = une colonne de plus, jamais une relance.
7. **Un jeu apparié construit pour la grille : 100 prompts de base en anglais, 20 par ancre, traduits en français, russe
   et chinois, soit 400 prompts** (validé par l'utilisateur le 2026-10-28). Raison : le jeu de test ne remplit pas une
   grille langue × palier (le français publiable n'a aucun `high` et aucun `needs_tools`, § 1), et la règle 5 de la
   rubrique (une traduction a les mêmes labels) rend l'écart entre langues mesurable à contenu identique.
8. **Les 100 prompts de base sont réels, choisis dans un vivier ouvert de licence permissive**, jamais écrits par un
   modèle : Dolly-15k pour le bas de l'échelle, OASST2 (hors de nos splits) pour le milieu, Arena-Hard-auto pour le haut
   (§ 3.1). **Jamais lmsys, Magpie, WildChat, ni une capture client ou communautaire.** Raison : coller un prompt dans un
   chat, c'est l'envoyer à un tiers ; docs/01 § 6 et docs/25 interdisent le reste ; un jeu synthétique mesurerait la grille
   sur des prompts que personne n'a écrits. **Seule exception : la case `expert`**, que le vivier ne remplit pas : elle
   est faite des dix prompts écrits à la main de l'annexe A, relus par l'utilisateur, et reste à dix (plus petite que les
   autres, ce que le rapport écrit). Chaque prompt porte son `origin` (`pool` ou `handwritten`) et ses auteurs dans les
   métadonnées : la provenance reste auditable sans classer ces dix comme artificiels.
9. **Deux chiffres pour s1d1, toujours** : contre la médiane de tous les juges, et contre la médiane **sans** Gemma 4.
   Raison : s1d1 a appris Gemma 4 ; l'écart entre les deux dit s'il a appris la grille ou le juge.
10. **Les concurrents passent sur les mêmes prompts, sans clé** : Laya (question `tier` de LiteLLM), Clef-flash
    (question `score` avec la rubrique en critères), classifieur NVIDIA (anglais seul), témoins aléatoire et longueur.
    Chaque juge est aussi un classifieur : noté contre la médiane des autres (un-contre-le-reste).
11. **Un dépôt public à part, `s1d1-bench`** : kit, prompts, fichiers de juges, résultats, rapport, et le harnais RouteLLM
    de docs/27 qui y déménage. La rubrique et les docs de décisions restent ici. Code Apache-2.0, données CC-BY-4.0.
12. **Aucune revendication publique avant** un rapport complet avec au moins trois juges locaux et les témoins.

## 1. Ce qui a été vérifié

- **Jeu de test** (`dataset/final/test.parquet`, lu) : 10 000 lignes ; par source : aya 3 200, wildchat 2 000, lmsys
  1 667, ultrachat 1 333, magpie 1 000, oasst2 800. Paliers du juge d'entraînement : low 6 450, medium 2 506, trivial 718,
  high 217, null 106, expert 3. **Sans lmsys ni magpie : 7 333 lignes, 65 langues**, high 136, expert 1, null 65.
  Conséquence : tiré du jeu de test, un banc ne pourrait pas mesurer `expert`, et mesurerait `high` sur 136 prompts au plus.
- **Par langue, sources publiables du test** (lu le 2026-10-28) : anglais 2 165 lignes (trivial 67, low 1 254, medium
  755, high 59, expert 1, `needs_tools` vrais 89) ; français 123 (trivial 6, low 72, medium 43, **high 0, `needs_tools` 0**) ;
  russe 175 (trivial 5, high 2, `needs_tools` 8) ; chinois 261 (trivial 9, high 31, `needs_tools` 8). Le train en a un peu
  plus (français high 28), mais ces prompts ont servi à l'entraînement : les mettre dans le banc serait une fuite.
  **Un banc par cases exige donc ses propres prompts** (décision 7).
- **Viviers de prompts réels, ouverts** (cartes lues le 2026-10-28, § 8 pour ce qui reste à relire) :

  | jeu | réel ? | licence | langues | paliers couverts | retenu |
  |---|---|---|---|---|---|
  | OASST2 | oui, bénévoles | Apache-2.0 | EN, ES, RU, DE, FR, ZH… | low, medium | oui, hors de nos splits |
  | Arena-Hard-auto (via `CohereLabs/m-ArenaHard`) | oui, utilisateurs de l'Arena, filtrés durs | Apache-2.0 | EN + 22 langues traduites par Google Translate | high, un peu d'expert ; codage et maths surreprésentés | oui, pour le haut ; ses traductions = témoin |
  | Dolly-15k | oui, employés Databricks | CC-BY-SA 3.0 | EN | trivial, low | oui, pour le bas |
  | WildBench v2 | oui, WildChat, 1 024 tâches | AI2 ImpACT low risk (conditions à propager) | EN | medium, high | non : licence à propager |
  | No Robots | oui, annotateurs | CC-BY-NC-4.0 (non vérifié) | EN | low, medium | non : non commercial |
  | Aya | oui, bénévoles | Apache-2.0 | 65 langues | trivial, low | déjà dans notre jeu ; non |
- **Ancres de score** (carte du Hub, `release/s1d1-router/README.md`) : trivial 0,01, low 0,25, medium 0,50, high 0,75,
  expert 1,00 ; palier prédit = ancre la plus proche. Le QWK publié est calculé ainsi. Le banc reprend cette règle.
- **Rubrique** : `docs/judge-rubric-v7.txt`, sortie « un objet JSON par prompt » ; règle 5 : la langue ne change pas les
  labels ; `complexity` peut être `null`. La consigne pour un chat doit juger **un lot**, donc changer la seule règle de
  format (§ 3.2) : c'est une variante `v7-batch`, versionnée.
- **Bruit de s1d1** (docs/18, trois graines de `small`) : σ de 0,004 à 0,006 sur QWK, exactitude, F1 et AUROC. Toute
  différence entre variantes sous 2 σ est du bruit ; le rapport trace cette bande.
- **Par langue** (docs/17) : QWK complexité de 0,36 (néerlandais) à 0,81 (chinois) sur le test, avec la part de `low`
  qui explique une partie de l'étalement. Le banc rapporte par langue avec la même mise en garde.
- **Ce qui existe déjà** : la rubrique ; le format parquet et la jointure par `prompt_key` (docs/07, docs/25) ; QWK,
  MAE, F1, AUROC et stats par langue (`pipeline/09`, docs/17) ; adaptateurs `bench/s1d1_adapter.py` et
  `bench/laya_adapter.py` ; témoins et IC par rééchantillonnage (`bench/routellm_eval.py`). **Ce qui manque** : un
  exécuteur de juges **local** (aujourd'hui `03_label_job.py` tourne sur HF Jobs avec vLLM) ; l'export de la consigne ;
  l'import et la validation des fichiers de juges ; le rapport avec graphiques.
- **Dépendances actuelles** (`pyproject.toml`) : pyarrow, datasets, huggingface-hub, transformers, torch, scipy,
  scikit-learn. Aucune bibliothèque de graphiques.
- **Hub** : `scombat/s1d1-router-bench` existe, dataset **privé, vide** (créé 2026-10-13). Réutilisable comme miroir
  des fichiers de juges et des résultats.
- **Clef-flash** (carte Hugging Face `Cloudflare/clef-flash`, lue 2026-10-28) : Apache-2.0, Qwen3.5-9B + tête de schéma
  jointe, questions `noul` / `choice` / `score` avec critères, pas de génération ; chargeur maison (`joint_schema_model.py`),
  testé avec transformers 5.10.2 ; `max_length` 16 384 jetons ; médiane 38,8 ms sur H200. Tourne en principe sur le Mac
  (9B, bf16 ≈ 18 Go) ; non essayé.
- **Cloudflare Auto Router** (billet du 2026-09-30, doc, InfoQ) : classifieur multi-têtes non nommé, 14 catégories,
  quatre notes de 1 à 5 (complexité, ambiguïté, enjeu, dépendance au contexte) ; banc de 97 tâches internes × 3
  échantillons, IC bootstrap ± 6 à 7 points, aucun témoin, aucun routeur concurrent, Opus 5.5 et GPT-6 Sol en opt-in
  seulement. C'est le contre-exemple que ce document évite.

## 2. Ce que le banc mesure, et ce qu'il ne mesure pas

**Mesuré.** Pour chaque classifieur, l'accord de ses quatre sorties avec la médiane des juges : palier de complexité
(QWK, exactitude, exactitude à ± 1 palier, matrice de confusion), score (MAE contre l'ancre du palier médian, courbe de
calibration : parmi les prompts notés 0,70 à 0,80, quelle part les juges mettent-ils en `high` ou au-dessus ?), tâche
(exactitude, F1 macro, confusions entre voisines : editing / summarization / extraction, chat / information_seeking),
`needs_tools` (F1, AUROC, sur peu de positifs : l'IC le dira). Le tout globalement, par langue, par source, par palier.

**Plafond et plancher.** Plafond = accord de chaque juge avec la médiane des autres ; un classifieur au-dessus imite un
juge, il n'est pas meilleur que la grille. Plancher = témoins aléatoire (10 tirages) et longueur du prompt (en caractères,
seuils par quantiles). Entre les deux, le bruit des graines de s1d1 (docs/18) en bande.

**Non mesuré, à écrire dans le README du dépôt.** (a) Si la grille elle-même correspond à « quel modèle suffit » : c'est
l'autre banc, celui des réponses, laissé de côté par décision de l'utilisateur (conversation du 2026-10-28). (b) Le routage
(mapping score → modèle) : affaire du déployeur. (c) La distribution réelle des prompts : le jeu est construit par cases
(§ 3.1), et `expert` sera la case la moins fournie. (d) Les prompts agentiques et multi-tours : le jeu est fait de
premiers prompts. (e) Les langues autres que les quatre choisies : docs/17 reste la seule mesure sur 78 langues.

## 3. Le kit en trois commandes

### 3.1 Le jeu apparié, et `export`

**Construction du jeu (une fois, à la main, documentée).**

1. *Vivier* : Dolly-15k, OASST2 (premiers messages, toutes langues, **moins** les `prompt_key` présents dans
   `dataset/final/*` et dans `dataset/interim`, par le filtre `--exclude` de docs/25), Arena-Hard-auto v0.1 en anglais.
   Le vivier complet est publié dans le dépôt avec sa source et sa licence par ligne.
2. *Pré-tri* : un juge local (pas Gemma 4, pour ne pas cumuler les biais) étiquette quelques centaines de candidats
   anglais avec la rubrique `v7-single`. Les pré-tris sont publiés aussi.
3. *Tirage* (amendé le 2026-10-29 ; la version du 2026-10-28 disait « choix à la main par l'utilisateur ») : **tirage
   aléatoire stratifié à graine fixe** sur les étiquettes du pré-tri : 20 prompts par strate `trivial`, `low`, `medium`,
   `high` (80 du vivier), dont un quota pris dans la strate `needs_tools` vrai pour en avoir au moins 20 au total, et un
   tirage par tâche quand une tâche est sous 4. Personne ne choisit un prompt. Le seul geste manuel est un **veto borné
   à deux règles** : deux prompts du même gabarit, et un prompt dont la traduction changerait la tâche (grammaire
   anglaise, jeu de mots) ; l'assistant prépare la liste des cas, l'utilisateur confirme, chaque retrait est noté avec
   sa raison et le prompt suivant de la strate le remplace. Pour `expert`, le vivier n'est pas utilisé : la case est faite
   des dix prompts de l'annexe A (`origin: handwritten`), et reste à dix, soit 90 prompts de base et 360 lignes (le
   « 400 » de ce document est un arrondi). La composition porte le biais du juge de pré-tri, nommé et publié ; la vérité
   du banc reste la médiane des juges, calculée après, et le rapport classe chaque prompt dans la case de cette médiane,
   pas dans sa strate. Règle, graine, clés tirées et retraits sont publiés pour que le tirage soit rejouable.
4. *Traduction* : les 100 prompts traduits en français, russe et chinois par un modèle local, puis **relus** (l'utilisateur
   pour le français ; pour le russe et le chinois, un juge local en relecture, puis la communauté après ouverture). Pour
   les prompts venant d'Arena-Hard, les traductions Google de m-ArenaHard servent de témoin, pas de référence. Un prompt
   dont la traduction change la tâche (par exemple un prompt sur la grammaire anglaise) est remplacé.
5. *Fichier* : 400 lignes, `id` = hachage du prompt de base + code de langue, pour que les quatre versions se rejoignent
   dans `report` sans que le juge le voie.

**Ce que ce jeu mesure et ne mesure pas.** Il mesure la fidélité à la grille à contenu contrôlé, et l'écart entre langues
à contenu identique. Il ne mesure pas la distribution réelle des prompts : c'est le rôle du jeu de test de docs/17, qui
reste publié à côté. Vingt prompts par case ne donnent pas de chiffre par case lisible (± 0,3 de QWK) : le rapport publie
les marges, par langue (100), par palier (80), par tâche, jamais la case.

**`export`**, régénérable à l'identique (ordre mélangé par graine fixée) :

- `prompts/<jeu>.jsonl` : une ligne par prompt, champs `id`, `prompt`, et rien d'autre. Ni source, ni langue, ni palier
  visé, ni aucune prédiction : les juges ne doivent rien savoir (tout cela vit dans `prompts/<jeu>.meta.parquet`, lu
  seulement par `report`).
- `prompts/<jeu>.lots/NN.md` : les mêmes prompts par lots de 40, prêts à coller dans un chat, chaque lot précédé de la
  consigne, les quatre langues mélangées. Taille de lot à ajuster au premier essai (§ 7, question 5).
- `consigne/v7-batch.md` : la rubrique v7 **mot pour mot**, puis la seule règle changée : « pour chaque prompt du lot,
  une ligne `{"id": "<id>", "complexity": …, "task": …, "needs_tools": …}`, rien d'autre ». Les identifiants sont des
  hachages : ils n'apprennent rien au juge.
- `consigne/v7-single.md` : la rubrique inchangée, pour l'exécuteur local qui juge un prompt à la fois, comme
  `03_label_job.py`. L'écart entre juger seul et juger en lot est mesuré une fois (§ 4.3).

### 3.2 Le fichier de juge

Un dossier par juge, `judges/<nom>/` :

- `meta.json` : `judge` (nom libre), `model` (identifiant exact), `provider` (`local`, `chat`, `api`, `human`),
  `revision` (commit ou version si connue), `date`, `temperature` et `thinking` (si connus ; `"unknown"` sinon),
  `instructions` (`v7-batch` ou `v7-single`), `prompts_sha256` (empreinte du fichier de prompts jugé), `author`,
  `replayable` (`true` seulement pour `local` à température 0, révision figée), `licence` (celle sous laquelle
  l'auteur verse le fichier, § 6), `notes`.
- `labels.jsonl` : une ligne par `id`, les trois champs de la rubrique. `complexity` peut valoir `null`.

**Règle d'acceptation** (appliquée par `run`, et par la CI du dépôt sur chaque PR) : `meta.json` complet ;
`prompts_sha256` égal à celui d'un fichier de prompts du dépôt ; tous les `id` présents, aucun en trop, aucun doublon ;
valeurs dans les énumérations de la rubrique ; sinon rejet avec la liste des lignes fautives. Pas de contrôle de qualité
du contenu : un juge médiocre se verra dans son accord avec les autres, c'est le rapport qui le dit.

**Immutabilité** : un fichier accepté ne change plus. Correction humaine = nouveau dossier `judges/<nom>-corrige-<auteur>/`
avec `provider: human`, `notes` disant ce qui a été corrigé. Les deux apparaissent dans le rapport.

### 3.3 `run` : fusion des juges, puis les classifieurs

1. Charge tous les dossiers `judges/` valides ; calcule par prompt la **médiane** des paliers (échelle ordinale
   trivial < low < medium < high < expert ; `null` compté à part : un prompt est « indécidable » si la majorité des juges
   dit `null`), le vote majoritaire pour `task` et `needs_tools`, et l'**écart** par prompt (nombre de paliers entre le
   juge le plus bas et le plus haut). Médiane « tous » et médiane « sans Gemma 4 ». Deux médianes par prompt de base :
   **par langue** (K votes) et **toutes langues** (4 × K votes, la vérité principale, puisque la règle 5 dit que les
   quatre versions ont les mêmes labels) ; un prompt où la médiane par langue s'écarte de la médiane toutes langues est
   signalé comme désaccord de langue (grille, traduction, ou juge).
2. Fait tourner chaque classifieur sur `prompts/<jeu>.jsonl`, en local, sans clé : les variantes s1d1 (full, compact,
   small, tiny ; int8 et fp16 ; les trois graines de `small`), Laya english et multilingual (question `tier`, critères
   de LiteLLM, comme docs/27), Clef-flash (question `score` à cinq options, critères = les cinq définitions de la
   rubrique ; `task` en `choice` à 19 options ; `needs_tools` en `noul`), classifieur NVIDIA (anglais seul, dimension
   agrégée à définir), témoins. Chaque classifieur écrit `classifiers/<nom>/labels.jsonl` + `meta.json` au **même format**
   qu'un juge (plus `score` pour ceux qui en ont un).
3. Produit `results/<jeu>.parquet` : une ligne par prompt, une colonne par juge et par classifieur pour chaque sortie,
   plus médianes et écarts. C'est le fichier qu'on ajoute, jamais qu'on recalcule de zéro : un dossier nouveau = ses
   colonnes nouvelles.

### 3.4 `report` : tableaux, graphiques, données brutes

- Tableaux (Markdown et CSV) : accord de chaque classifieur et de chaque juge avec la médiane (tous / sans Gemma 4 /
  un-contre-le-reste pour les juges), avec IC 95 % par rééchantillonnage des prompts (1 000 tirages) ; par langue, par
  source, par palier.
- Graphiques (SVG, un fichier par figure) : barres QWK avec IC, plafond (juges) et plancher (témoins) en lignes, bande
  des graines ; matrices de confusion ; courbe de calibration du score ; écart entre juges par palier ; QWK par langue.
- `report/index.md` généré, avec les figures et les mises en garde du § 2 recopiées depuis un gabarit. Données brutes
  à côté pour que chacun refasse ses propres figures.
- Bibliothèque de graphiques : **matplotlib**, dans le dépôt du banc seulement (décision de l'utilisateur). Analyses
  à produire, toutes avec IC par rééchantillonnage : accord ordinal (QWK, alpha de Krippendorff ordinal, exactitude à
  ± 1), accord binaire (F1, AUROC, MCC, courbe précision-rappel pour `needs_tools`), tâche (exactitude, F1 macro, matrice
  de confusion normalisée), score (MAE, Spearman, diagramme de fiabilité et erreur de calibration attendue sur dix cases
  de score), comparaison appariée entre classifieurs (IC bootstrap de la différence sur les mêmes prompts), et par langue
  l'écart à contenu identique (le même prompt de base dans ses quatre versions). Chaque figure a son CSV à côté.

## 4. Les juges

### 4.0 Taille

400 prompts (100 × 4 langues), au lieu des 1 000 de la première version de ce document : ≈ 240 000 jetons de sortie par
juge, **≈ 3 h par juge local de 27 à 31B**, et 10 lots de 40 pour une contribution par chat.

### 4.1 Ligne de base rejouable, locale

Candidats à poids ouverts tournant sur le Mac (128 Go), **tailles et licences à vérifier avant choix** : Gemma 4 31B-it
(le juge de l'entraînement, consigne et plafond de réflexion de docs/05 : 512), Qwen3.8 27B, gpt-oss-20b, GLM 5.2,
DeepSeek V4 flash. Trois au minimum, dont Gemma 4. Exécution : MLX (8 bits) ou llama.cpp, température 0, un prompt à la
fois (`v7-single`), révision du modèle et du quantifieur dans `meta.json`. Un juge local est rejouable **au quantifieur
près** : changer de quantification = nouveau juge.

Ordre de grandeur : 400 prompts × ≈ 600 jetons de sortie (réflexion plafonnée comprise) = 240 000 jetons par juge ;
à 25 jetons/s pour un 27 à 31B en 8 bits, **≈ 3 h par juge**, moins pour un 20B. À mesurer sur 50 prompts avant de lancer.

### 4.2 Contributions

ChatGPT, Claude, Gemini, un humain : même format, `replayable: false`, lots de la consigne `v7-batch`. **Le chiffre
principal est la médiane des juges locaux seuls** (décision de l'utilisateur) ; les contributions figurent en annexe du
rapport, chacune contre la médiane locale, et une médiane « tous juges » est donnée à titre indicatif.

### 4.3 Deux effets à mesurer une fois, sur 100 prompts

- **Seul contre lot** : le même juge local en `v7-single` et en `v7-batch` ; si l'accord entre les deux est sous le
  plafond inter-juges, les contributions par chat ne sont pas comparables aux juges locaux et le rapport les sépare.
- **Réflexion** : le juge local avec et sans plafond de réflexion (docs/05 avait tranché 512 pour le coût ; ici le coût
  est du temps).

### 4.4 Circularité

Gemma 4 a produit les étiquettes d'entraînement. Trois garde-fous : jamais seul ; les deux chiffres de la décision 9 ;
et l'accord de Gemma 4 avec la médiane des autres, publié comme n'importe quel juge. Si s1d1 est nettement plus proche
de Gemma 4 que de la médiane sans lui, le README le dit en une phrase.

## 5. Ce que le rapport doit permettre de dire

Un chiffre du type : « sur N prompts réels en L langues, s1d1 small (int8, 92 Mo) s'accorde avec la médiane de K juges à
QWK x [IC], le plafond inter-juges est y, Laya z, Clef-flash w, la longueur du prompt v ». Et sa version négative, si
c'est ce qui sort. **Aucune phrase « meilleur que »** sans que l'IC de l'écart exclue zéro et dépasse 2 σ des graines.

## 6. Le nouveau dépôt

- **Contenu** : le kit (`export`, `run`, `report`, tests), `prompts/`, `consigne/`, `judges/`, `classifiers/`,
  `results/`, `report/`, et `routellm/` (le harnais de docs/27, déplacé tel quel, mêmes résultats). CI : validation des
  fichiers de juges sur chaque PR, tests, régénération du rapport.
- **Reste ici** : la rubrique (source de vérité, copiée dans le kit avec son numéro), les docs de décisions, le pipeline.
  Le dépôt du banc **n'importe pas** le dépôt de recherche : s1d1 est chargé comme n'importe quel classifieur, par ONNX
  depuis le Hub.
- **Licences** : code Apache-2.0 (comme tantive et alliance). Prompts : la licence de chaque source s'applique et doit
  être relue sur sa carte avant publication (§ 8). Fichiers de juges, prompts écrits à la main, résultats et rapport :
  **CC-BY-4.0** (décision de l'utilisateur), et une ligne dans `meta.json` qui dit sous quoi le contributeur verse son
  fichier (CC-BY-4.0 par défaut, exigé pour être fusionné).
- **Visibilité** : privé jusqu'au premier rapport avec trois juges locaux et les témoins ; public ensuite, dans le
  lancement par étapes. Miroir des données sur `scombat/s1d1-router-bench` (Hub), qui passe public au même moment.
- **Dépendances** du dépôt : pyarrow, onnxruntime, tokenizers (comme `S1D1Router` de docs/27 § 5), numpy ; par
  classifieur tiers, un venv à part comme `bench/setup.sh` (Laya, Clef-flash avec transformers 5.10, NVIDIA) ; l'exécuteur
  de juges, mlx-lm ou llama.cpp, dans son venv. Rien de tout cela dans `pyproject.toml` du dépôt de recherche.
- **Nom** : **`s1d1-bench`** (décision de l'utilisateur). Créé au début du premier lot de code, privé d'abord ; sa
  création est une étape du lot, pas de ce document.

## 7. Questions posées et réponses de l'utilisateur (2026-10-28)

1. **Décisions 1 à 12** : relues, validées.
2. **Dépôt** : `s1d1-bench`, à créer (au premier lot de code).
3. **Vivier** : Dolly-15k + OASST2 + Arena-Hard-auto, les trois.
4. **`expert`** : les dix prompts de l'annexe A, relus par l'utilisateur, sans les classer comme synthétiques ; pas de dix
   autres tirés du vivier. Appliqué : champ `origin` (`pool` / `handwritten`) à la place d'une marque « synthétique »,
   case `expert` à dix.
5. **Juges locaux** : Gemma 4 31B-it, Qwen3.8 27B, gpt-oss-20b, un quatrième si le temps le permet.
6. **Graphiques** : matplotlib, avec l'ensemble des analyses du § 3.4.
7. **Contributions par chat** : médiane locale seule comme chiffre principal, contributions en annexe.
8. **Licence des données** : CC-BY-4.0.
9. **Fichier humain de l'utilisateur** : non, pas indispensable vu le protocole.

## 8. Non vérifié

- Les licences du vivier pour la **redistribution des prompts** dans un dépôt public, relues sur les cartes officielles
  et non sur des miroirs : Dolly-15k (CC-BY-SA 3.0, une copie dit CC-BY 3.0), OASST2 (Apache-2.0 selon docs/01, à
  confirmer), Arena-Hard-auto v0.1 (le dépôt `lmarena/arena-hard-auto` est Apache-2.0 ; les prompts viennent de l'Arena,
  dont les conversations sont sous une licence à part : vérifier que le jeu de questions est bien couvert par Apache) et
  m-ArenaHard (Apache-2.0 selon sa fiche).
- La taille du vivier OASST2 **hors de nos splits** par palier, et combien de `expert` réels existent dans Arena-Hard
  (décision 8, question 4).
- Que Clef-flash tourne sur le Mac avec son chargeur maison, et que ses questions `score` à critères acceptent un texte de
  rubrique aussi long. Que la licence « NVIDIA Open Model » permette d'exécuter et de publier des résultats du
  classifieur NVIDIA.
- Les tailles, licences et quantifications disponibles de Qwen3.8 27B, gpt-oss-20b, GLM 5.2, DeepSeek V4 flash ; le
  débit réel en jetons/s sur le Mac (l'estimation de 7 h par juge est un ordre de grandeur).
- Le comportement des interfaces de chat sur des lots de 40 prompts multilingues (longueur, refus sur les prompts
  nuisibles de la règle 8, format respecté) : à tester sur un lot.
- La taille de lot et le nombre de messages que représente un fichier complet (10 lots de 40 pour 400 prompts).
- La qualité des traductions par un modèle local vers le russe et le chinois, et si un juge local suffit à les relire.

## Annexe A — Dix prompts `expert` écrits à la main (2026-10-28)

Écrits en conversation par l'utilisateur et Claude, relus par l'utilisateur, **pas tirés du vivier** : dans
`prompts/<jeu>.meta.parquet` ils portent `origin: handwritten` et `author: "scombat + Claude"`, les autres `origin: pool`
avec leur source. Ils forment à eux seuls la case `expert` (§ 7, réponse 4).
Dix domaines, sept tâches, quatre `needs_tools` vrais (1, 2, 4, 5). Les étiquettes ci-dessous sont **celles visées par les
auteurs**, pas une vérité : ce sont les juges qui diront si la grille les met à `expert`. Si deux juges sur trois disent
`high`, c'est la grille qui parle. Leur traduction en russe et en chinois est la plus délicate du jeu (terminologie).

### A.1 Mathématiques — `information_seeking`, `needs_tools: true`

> Is the Brauer–Manin obstruction known to be the only obstruction to the Hasse principle for diagonal quartic surfaces
> over Q? Check the literature from 2024 onward (arXiv and published) for any new unconditional result, then give the
> strongest unconditional and conditional results with what each assumes, and one explicit surface where the question is
> still open.

### A.2 Systèmes concurrents — `coding_debugging`, `needs_tools: true`

> This bounded MPMC queue passes tests on x86 but corrupts data on an ARM server. Write a herd7 litmus test that exhibits
> the forbidden outcome under the C++20 memory model, run it, then fix the queue with the weakest orderings that are still
> correct and show the litmus test no longer finds the outcome.
>
> ```cpp
> struct Cell { std::atomic<size_t> seq; T data; };
> bool push(const T& v) {
>   size_t pos = tail.load(std::memory_order_relaxed);
>   for (;;) {
>     Cell& c = buf[pos & mask];
>     size_t seq = c.seq.load(std::memory_order_relaxed);
>     intptr_t d = (intptr_t)seq - (intptr_t)pos;
>     if (d == 0) { if (tail.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) { c.data = v; c.seq.store(pos + 1, std::memory_order_relaxed); return true; } }
>     else if (d < 0) return false;
>     else pos = tail.load(std::memory_order_relaxed);
>   }
> }
> ```

### A.3 Pharmacologie clinique — `planning`, `needs_tools: false`

> 67-year-old, CKD stage 4 (eGFR 22 mL/min), on apixaban 5 mg twice daily for atrial fibrillation, now diagnosed with
> invasive pulmonary aspergillosis and starting voriconazole. Propose a dosing and monitoring plan for the next four weeks
> that handles the CYP3A4 interaction, says when to draw voriconazole troughs and what targets to use, and states the one
> situation in which you would switch the anticoagulant.

### A.4 Droit des données — `reasoning`, `needs_tools: true`

> A controller in Ireland uses a processor in the UK that sub-processes in India. Map the transfer mechanism required at
> each hop under GDPR and UK GDPR today, including the transfer impact assessment expected after Schrems II, and explain
> exactly which hop breaks and what replaces it if the EU adequacy decision for the UK is not renewed.

### A.5 Cryptographie — `reasoning`, `needs_tools: true`

> Look up the current version of draft-ietf-tls-hybrid-design and its exact key schedule. A hybrid KEM concatenates the
> ML-KEM-768 and X25519 shared secrets and feeds them to HKDF-SHA256 with both ciphertexts in the info field. (a) State
> the precise condition under which this combiner stays IND-CCA secure when one component is fully broken, and name the
> 'dual-PRF' requirement on the KDF. (b) Explain what ML-KEM-768 decapsulation actually does with a malformed or tampered
> ciphertext: how the FO transform with implicit rejection derives the output from the secret value z and the ciphertext
> hash instead of returning an error, and why the two-stage check (re-encryption and comparison) must be constant-time.
> (c) Show how, in this combiner, implicit rejection plus binding the ciphertexts into the KDF info closes the
> key-mismatch and chosen-ciphertext rejection attacks that an explicit-rejection KEM would expose, and name the one
> attack it does not close. (d) Say whether the draft's current construction meets all of this, citing the section.

### A.6 Calcul de structure — `math`, `needs_tools: false`

> IPE 400 in S355, simply supported 8 m span, no intermediate lateral restraint, end moments giving a linear diagram with
> ψ = −0.5. Compute Mb,Rd to Eurocode 3 with the general method (6.3.2.2), using the correct C1 for that moment shape, and
> state whether the simplified method of 6.3.2.3 would be admissible here and what it would give.

### A.7 Linguistique historique — `information_seeking`, `needs_tools: false`

> Reconstruct the Proto-Indo-European word for 'wheel' from Greek kýklos, Sanskrit cakrá-, Tocharian B kokale and Old
> English hwēol, explain why the reduplication is original rather than secondary, and say what the Tocharian form tells
> us that the others cannot.

### A.8 Dérivés de taux — `reasoning`, `needs_tools: false`

> For a EUR/USD cross-currency basis swap with mark-to-market notional resets on the USD leg, under a USD cash CSA: which
> curve discounts each leg, how does the valuation differ from the constant-notional swap, and where does the
> cross-currency basis enter the EUR discount curve construction? Derive it, don't just state it.

### A.9 Archéométrie — `writing`, `needs_tools: false`

> Write the Methods section (about 400 words) for a Bayesian chronological model of a stratified Neolithic tell with 42
> radiocarbon dates, as a reviewer at Radiocarbon would expect it: calibration curve and its version, marine and
> freshwater reservoir corrections for the fish-eating individuals, the outlier model you use and why, how the
> stratigraphic phases enter the model as priors, and how you report agreement indices and a model that fails the
> threshold.

### A.10 Sismologie — `data_analysis`, `needs_tools: false`

> Five stations recorded a local earthquake. Station (x km, y km): P arrival, S arrival, in seconds after a common
> reference. A (0, 0): 4.12, 7.31. B (30, 0): 5.90, 10.45. C (0, 40): 6.85, 12.10. D (30, 40): 7.60, 13.42.
> E (15, 20): 3.95, 7.00. Assume Vp = 6.0 km/s and Vp/Vs = 1.73. Estimate the origin time, epicenter and depth by hand
> using S-minus-P distances, say which of the four unknowns is worst constrained by this geometry and why, and explain
> how a 5 % error in Vp/Vs would move your depth estimate.
