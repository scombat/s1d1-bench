# 33 — Lot 2 du banc : traduction du jeu de base en français, russe et chinois (2026-10-29, validé)

Document de décisions, **rien n'est codé**. Validé par l'utilisateur le 2026-10-29 avec les avis donnés en § 5,
puis amendé le soir même (exécution sans arrêt humain, voir la fin du § 5).
Suite de docs/32 (banc de fidélité de la note, § 3.1 étape 4 : « les 100 prompts traduits en français, russe et
chinois par un modèle local, puis relus ») et de son amendement du 2026-10-29 (tirage stratifié). État au
2026-10-29 : `s1d1-bench` a fusionné le lot 1 (kit, vivier, juges, essai) et le pré-tri (`bench draw`, jeu `base-v1`
figé). Ce document fixe comment le jeu anglais devient le jeu apparié de 360 lignes, et ce que le lot 2 code.

## 0. Décisions proposées

1. **Le traducteur est Qwen3.8 27B (8 bits, MLX), sans réflexion, température 0, révision figée**, le même que le
   pré-tri. Raison : pas Gemma 4, pour que le juge d'entraînement de s1d1 ne façonne pas les textes que les juges
   notent ensuite (même raison que pour le pré-tri) ; pas de modèle de traduction dédié, qui serait une dépendance et un
   téléchargement de plus sans preuve qu'il fasse mieux sur des prompts de chat (code, consignes, passages collés).
   Si la relecture (§ 3) montre que Qwen3 ne tient pas sur une langue, on change de modèle pour cette langue, et on
   l'écrit.
2. **Une consigne de traduction fixe, publiée, versionnée** (§ 2.2) : traduire le message de l'utilisateur tel quel,
   même tâche, même registre, même longueur, sans répondre, sans corriger, sans rien ajouter ni retirer ; le code,
   les identifiants, les URL, les nombres, les noms propres et les chaînes citées restent tels quels ; les passages
   collés (Dolly) sont traduits aussi, parce que la tâche est posée sur eux.
3. **Les sorties du modèle ne sont jamais modifiées.** Une correction humaine est un fichier de recouvrement à part
   (`<lang>.corrections.jsonl` : id, texte corrigé, raison, auteur), appliqué par l'export. Raison : la même règle que
   pour les fichiers de juges ; on garde ce que le modèle a produit et ce que l'humain a changé, séparément et
   rejouable.
4. **Relecture du français par l'utilisateur, intégrale** : 90 prompts, 37 000 caractères. Les corrections passent par
   le fichier de recouvrement, jamais par une retouche du fichier du modèle.
5. **Relecture du russe et du chinois par deux vérifications locales, puis la communauté** : (a) vérification
   structurelle automatique (§ 3.2) ; (b) **même tâche** : Qwen3 étiquette chaque traduction avec la rubrique v7 et
   l'on compare `task` et `needs_tools` à son étiquette du pré-tri anglais ; (c) **même demande** : Gemma 4, l'autre
   juge, reçoit l'anglais et la traduction et répond si la demande est la même, rien d'ajouté, rien de perdu, rien de
   répondu, avec un motif. Un écart sur (b) ou (c) est un signalement, lu par l'assistant (qui lit les trois langues),
   tranché par l'utilisateur.
6. **Les traductions Google de m-ArenaHard sont un témoin, pas une référence** (docs/32 § 3.1) : pour les 28 prompts
   Arena-Hard du jeu, la vérification (b) est aussi faite sur la version Google ; si notre traduction et la version
   Google donnent la même étiquette que l'anglais, le signalement est levé ; si elles divergent entre elles, l'assistant
   lit.
7. **Un prompt dont la traduction change la tâche est remplacé par la règle 2 du veto** (docs/32 § 3.1 amendé) : on
   ajoute l'id au fichier de veto avec sa raison, `bench draw` donne le suivant de sa strate, qui est traduit à son
   tour. Pas de retouche de la traduction pour « sauver » un prompt.
8. **Les dix prompts écrits à la main sont traduits par le même chemin**, relus en français par l'utilisateur comme
   les autres. `bench export` doit apprendre à les sortir dans les quatre langues (aujourd'hui anglais seulement).
9. **Codes de langue ISO 639-3** comme le reste du projet (docs/17, `pipeline/01_interim.py`) : `eng`, `fra`, `rus`,
   `zho` ; le chinois est le chinois simplifié, et le fichier le dit.
10. **Ce que le lot 2 livre** : `bench translate` (traduction d'un jeu de base dans une langue, fichier + métadonnées),
    `bench check` (les vérifications du § 3), l'extension de `bench export` aux traductions et aux prompts écrits à
    la main, et le jeu `data/prompts/base-v1.jsonl` de 360 lignes. **Pas** les juges sur les 360 lignes, pas `run`,
    pas `report`.

## 1. Ce qui a été vérifié (2026-10-29)

- **Jeu de base figé** (`data/prompts/base-v1.select.txt`, PR #3 fusionnée) : 80 prompts du vivier, par source OASST2
  34, Arena-Hard 28, Dolly 18 ; par strate et source : high = 16 Arena-Hard + 4 OASST2 ; medium = 11 Arena-Hard +
  6 OASST2 + 3 Dolly ; low = 11 OASST2 + 8 Dolly + 1 Arena-Hard ; trivial = 13 OASST2 + 7 Dolly. 32 065 caractères en
  tout, médiane 118, maximum 3 456 ; un seul prompt avec un bloc de code, un seul avec une URL. Plus les dix prompts
  écrits à la main (`data/handwritten/expert-v1.jsonl`, 4 902 caractères, dont un extrait C++).
- **`bench export`** (lu) : une ligne du vivier porte un champ `lang` (défaut `eng`) et l'export prend les lignes dont
  la langue est dans `--langs` ; l'`id` est `base_key-lang`. Les lignes traduites sont donc déjà prévues pour le vivier.
  Les prompts écrits à la main ne sortent qu'en `eng` (`if "eng" not in langs: continue`) : à étendre.
- **m-ArenaHard** (`CohereLabs/m-ArenaHard`, Apache-2.0, carte lue) a des configurations `fr`, `ru` et `zh` (traductions
  Google des 500 prompts d'Arena-Hard-auto), avec le même `question_id` que notre champ `source_id`. Les 28 prompts
  Arena-Hard du jeu ont donc une traduction témoin dans les trois langues.
- **Modèles présents sous `~/mlx-models/`** : Gemma 4 31B-it 8 bits, Qwen3.8 27B 8 bits, gpt-oss-20b. Aucun modèle de
  traduction dédié. Vitesses mesurées à l'essai (docs/32, `judges/TRIAL.md`) : 9,3 s par prompt pour Qwen3 sans
  réflexion avec une rubrique de 1 800 jetons en entrée et 21 jetons en sortie ; la traduction inverse le rapport
  (entrée courte, sortie de la longueur du prompt), la durée réelle est à mesurer sur dix prompts avant la série.
- **Règle 5 de la rubrique v7** : la langue ne change pas la note. C'est ce que le jeu apparié mesure ; il faut donc que
  la traduction change la langue et rien d'autre.

## 2. La traduction

### 2.1 Entrée, sortie, métadonnées

- Entrée : `data/prompts/base-v1.select.txt` + `pool.parquet` pour les 80, `data/handwritten/expert-v1.jsonl` pour
  les 10 ; une langue cible par exécution.
- Sortie : `data/translations/base-v1/<lang>.jsonl` (une ligne par prompt : `id` = `base_key-lang`, `base_key`,
  `prompt`) et `data/translations/base-v1/<lang>.meta.json` (modèle, révision, quantification, version de `mlx-lm`,
  température 0, réflexion coupée, `max_tokens`, SHA-256 de la consigne, SHA-256 du jeu anglais, date, auteur). Un
  `log.jsonl` comme pour les juges (jetons, durée, état par prompt). Même règle d'immuabilité que `judges/` :
  `bench translate` refuse d'écrire dans un dossier qui existe, `--resume` reprend une série interrompue.
- Appel : un prompt par génération, message système = la consigne, message utilisateur = le prompt entre balises
  `<prompt>` (le même dispositif que `bench judge`, pour que le texte soit une donnée et jamais une instruction) ;
  sortie = la traduction seule. Une réponse vide, tronquée (`finish_reason`), ou qui contient encore une balise
  `<prompt>`, est une erreur consignée, pas une traduction.

### 2.2 La consigne (à figer dans `consigne/translate-v1-<lang>.md`, versionnée, publiée)

Le fond, à écrire en anglais dans le fichier : tu traduis un message qu'un utilisateur a écrit à un assistant ;
traduis-le en `<langue>` comme si l'utilisateur l'avait écrit dans cette langue ; même demande, même ton, même niveau de
soin, même longueur ; ne réponds pas, ne commente pas, ne corrige pas les erreurs de fond, n'ajoute rien, n'omets rien ;
laisse tels quels le code, les commandes, les identifiants, les URL, les nombres, les unités, les noms propres, les
noms de produits et les chaînes entre guillemets ; traduis les passages collés comme le reste ; garde la mise en forme
(retours à la ligne, listes, blocs) ; rends la traduction seule. Pour le chinois : simplifié. Les fautes de frappe de
l'anglais n'ont pas d'équivalent : la consigne demande de rendre le niveau de soin (registre familier, ponctuation
négligée), pas la faute elle-même ; le rapport le dit.

### 2.3 Durée et ordre

270 générations (90 prompts × 3 langues), sortie d'environ 10 000 à 15 000 jetons par langue. À 10 à 15 jetons par
seconde en décodage, environ une heure pour les trois langues, à mesurer sur dix prompts avant. Ordre : français
d'abord (relu par l'utilisateur, donc le premier retour sur la consigne), puis russe et chinois avec la consigne
corrigée si besoin. Une consigne modifiée après le français est une nouvelle version (`translate-v2`) et le français est
refait avec elle : les trois langues sortent de la même consigne.

## 3. La relecture

### 3.1 Français : l'utilisateur (amendé le 2026-10-29 au soir : le contrôleur, voir § 5)

Les 90 prompts, d'une traite ou par lots de 40 (les fichiers `.lots/NN.md` de l'export servent à ça). Trois gestes
possibles par prompt : rien ; une correction (fichier de recouvrement `fra.corrections.jsonl`, raison en une ligne) ;
un veto de règle 2 (la traduction change la tâche) qui déclenche le remplacement. Le nombre de corrections et leur
nature (sens, registre, terme technique) sont écrits dans le rapport : c'est une mesure de la qualité du traducteur.

### 3.2 Vérification structurelle automatique (`bench check`, les trois langues)

Par prompt, la traduction contre l'anglais : même nombre de blocs ```, mêmes URL, mêmes suites de chiffres (dates,
quantités, versions), mêmes chaînes entre guillemets droits, même nombre de lignes à ±20 %, rapport de longueur dans
[0,7 ; 1,6] pour le français et le russe et [0,25 ; 0,9] en caractères pour le chinois ; aucune balise `<prompt>` ;
aucune ligne qui pourrait forger une enveloppe de lot (la même garde que `bench export`). Chaque écart est une ligne
de signalement (id, langue, règle), jamais une correction automatique.

### 3.3 Même tâche, même demande (`bench check`, russe et chinois ; français aussi, en plus de l'utilisateur)

- **Même tâche** : Qwen3 (le juge du pré-tri, sans réflexion) étiquette les 270 traductions avec la rubrique v7, par
  `bench judge` ordinaire (dossiers `judges/check-qwen3-off-<lang>`, qui sont des dossiers de juges comme les autres,
  publiés, et qui serviront aussi au banc). `task` ou `needs_tools` différents de l'étiquette anglaise du même
  prompt = signalement. Le palier n'est pas comparé : c'est précisément ce que le banc mesure, pas ce que la
  relecture doit forcer.
- **Même demande** : Gemma 4, sans réflexion, reçoit l'anglais et la traduction et répond en JSON `{same_request:
  bool, reason}` avec une consigne fixe (`consigne/check-same-request-v1.md`). `false` = signalement. Gemma 4 ne
  traduit rien et ne corrige rien : elle ne fait que lire.
- **Témoin Google** pour les 28 Arena-Hard : l'étiquette Qwen3 de la version m-ArenaHard de la même langue ; accord
  anglais = nôtre = Google lève le signalement de « même tâche » ; désaccord nôtre ≠ Google, l'assistant lit.
- Les signalements sont triés par l'assistant, qui lit les trois langues, dans une feuille par langue (id, règle,
  anglais, traduction, avis) ; l'utilisateur tranche chaque ligne : rien, correction, veto (amendé le 2026-10-29 au
  soir : le contrôleur tranche, l'utilisateur confirme à la PR, voir § 5). Après ouverture, la
  communauté relit par des fichiers de corrections à son nom (docs/32 § 4.2).

## 4. Fichiers, code, et ce que le lot 2 ne fait pas

- `consigne/translate-v1-fra.md`, `-rus.md`, `-zho.md` ; `consigne/check-same-request-v1.md`.
- `data/translations/base-v1/{fra,rus,zho}.jsonl`, `.meta.json`, `.log.jsonl` ; `.corrections.jsonl` quand il y en a.
- `data/translations/base-v1/witness-arena/{fra,rus,zho}.jsonl` : les 28 traductions Google, avec la licence et la
  révision de m-ArenaHard.
- `judges/check-qwen3-off-{fra,rus,zho}` (juges ordinaires sur les traductions) ; `data/checks/base-v1/<lang>.jsonl`
  (signalements) ; `data/checks/base-v1/<lang>.review.md` (feuille triée).
- `data/prompts/base-v1.jsonl` : 360 lignes, `.meta.parquet` avec `lang`, `origin`, `source`, `licence`, et pour les
  traductions `translator` (modèle, révision) et `corrected` (bool) ; `.sha256` ; lots de 40 pour les contributions par
  chat ; la consigne v7 inchangée.
- Code : `bench translate` (traduction, reprise, refus d'écraser), `bench check` (§ 3.2 et § 3.3, lecture des
  dossiers de juges et de Gemma 4 ; les générations elles-mêmes passent par le même `GenerateFn` que `bench judge`),
  `bench export --translations <dir>` (lignes traduites du vivier et des prompts écrits à la main, recouvrement des
  corrections, `translator` et `corrected` dans les métadonnées). Tests avec générations injectées, comme le lot 1.
- Hors lot 2 : le choix de gpt-oss comme troisième juge (toujours ouvert) ; les juges sur les 360 lignes (ils suivent
  le lot 2, environ 4 h Gemma 4 + 2 h 30 Qwen3 + gpt-oss ou remplaçant) ; `run` et `report` (lot 3) ; l'ouverture
  publique.

## 5. Points à trancher par l'utilisateur

Tranchés le 2026-10-29 : l'utilisateur a retenu l'avis donné à chaque point. Les cinq réponses font partie de
la décision ; les questions restent ci-dessous pour mémoire.

1. Qwen3 comme traducteur unique des trois langues (décision 1), ou Gemma 4 pour l'une d'elles si l'essai sur dix
   prompts est meilleur ? Mon avis : Qwen3 partout, et l'écrire ; changer seulement sur preuve.
2. Relecture du français intégrale (90 prompts) par toi, ou par lots avec un arrêt après le premier pour corriger la
   consigne ? Mon avis : le premier lot de 40 d'abord, puis la consigne figée, puis le reste.
3. Les corrections en français comptent-elles comme « traduction humaine » dans le rapport, ou le français reste-t-il
   « Qwen3 + corrections (n) » ? Mon avis : la seconde formule, avec le nombre.
4. Chinois simplifié seul (décision 9), ou simplifié et traditionnel ? Mon avis : simplifié seul, le jeu de test de
   docs/17 n'en distingue pas plus.
5. Les quatre dossiers de juges de vérification (Qwen3 sur fra, rus, zho) sont-ils aussi des juges du banc, ou
   seulement des outils de relecture ? Mon avis : des juges du banc, puisque ce sont les mêmes fichiers au même format
   et que Qwen3 est un des trois juges prévus ; cela économise 2 h 30.

**Amendement du 2026-10-29 au soir, validé par l'utilisateur** : le lot 2 s'exécute de bout en bout sans arrêt humain,
la relecture de la PR étant la seule porte. Deux points changent. (a) La relecture du français (§ 3.1, point 2
ci-dessus) est faite par le modèle contrôleur (Claude Fable 5.1), pas par l'utilisateur ; le rapport dit « Qwen3 +
n corrections par le contrôleur » et le champ `author` des fichiers de corrections le nomme. (b) Un veto de règle 2
n'est posé que sur deux signaux mécaniques réunis : l'étiquette `task` ou `needs_tools` de Qwen3 changée et non levée
par le témoin, et Gemma 4 qui répond `same_request: false`. Il est appliqué sur la branche avec sa raison, et
l'utilisateur le confirme à la relecture de la PR, où un veto refusé se défait avant fusion. Les feuilles de
relecture, les corrections et les vetos font partie de la PR.

## 6. Non vérifié

- La qualité de Qwen3.8 27B en russe et en chinois sur des prompts de chat : seule la relecture (§ 3) et la communauté
  le diront. Les chiffres du modèle sur des bancs de traduction n'ont pas été lus.
- La durée réelle de traduction (§ 2.3 est une estimation, à mesurer sur dix prompts).
- La stabilité des étiquettes Qwen3 entre un prompt et sa traduction : si la tâche change souvent pour des raisons de
  juge et non de traduction, le signalement « même tâche » sera bruyant ; on le saura au premier lot.
- Que les traductions Google de m-ArenaHard soient bien au même `question_id` que `source_id` dans notre vivier
  (déduit de la carte, pas encore joint).
