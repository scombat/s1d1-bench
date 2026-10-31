# Rapport du banc : {set_name}

Écrit par `bench report` depuis les résultats de `bench run` du {run_date}. Entrées : {inputs_line}. Ce fichier se
régénère : les tableaux sont dans `tables/` (CSV), les figures dans `figures/` (SVG, chacune avec son CSV). Le gabarit
est `report/TEMPLATE.md` ; ses trous sont remplis par le code, chiffres tels qu'ils sortent.

## Ce que ce rapport dit en premier

Deux lectures ont été écrites avant de voir les chiffres (docs/34 § 7.4) ; celle qui ne se vérifie pas est dite en
premier.

{predictions_first}

Tableaux : [`tables/predictions.csv`](tables/predictions.csv), [`tables/by_language.csv`](tables/by_language.csv) ;
dans cette page : [Les deux prédictions, en tableau](#les-deux-prédictions-en-tableau).

## La phrase de docs/32 § 5

> Sur {n_ids} ids, soit {n_base} prompts de base en {n_langs} langues, {headline_name} s'accorde avec la médiane de
> {k_judges} juges à QWK {headline_qwk} {headline_ci} ; le plafond inter-juges est {ceiling_mean} (chaque juge
> principal contre les deux autres, 8 votes et non 12 : {ceiling_values}) ; Laya english {laya_english}, Laya
> multilingual {laya_multilingual} ; l'heuristique de LiteLLM {litellm} ; la longueur du prompt {length}.
> L'IC de {headline_name} {position_vs_ceiling} ; {headline_name} {vs_floor}.

Tableaux : [`tables/agreement_all.csv`](tables/agreement_all.csv), [`tables/ceiling.csv`](tables/ceiling.csv),
[`tables/paired.csv`](tables/paired.csv) ; dans cette page : [Les candidats contre
`truth_all`](#les-candidats-contre-truth_all), [Le plafond](#le-plafond--chaque-juge-principal-contre-les-deux-autres),
[Les écarts appariés](#les-écarts-appariés).

{headline_others}

Tous ces chiffres sont des QWK contre `truth_all`, la médiane des votes des {k_judges} juges principaux sur les quatre
versions de chaque prompt de base, avec leur IC à 95 % : 1 000 tirages des {n_base} prompts de base, les quatre
versions suivent leur prompt de base. Clef-flash, nommé dans la phrase de docs/32, n'est pas mesuré dans ce lot
(docs/34, décision 4 : non vérifié, docs/32 § 8).

Les deux clauses de la fin sont choisies par des règles écrites dans le code (`src/s1d1_bench/report_index.py`) :

- position par rapport au plafond : « contient la moyenne du plafond » quand la moyenne du plafond tombe dans l'IC du
  chiffre principal ; « est au-dessous de la moyenne du plafond » quand la borne haute de cet IC est au-dessous de la
  moyenne ; « est au-dessus de la moyenne du plafond » quand sa borne basse est au-dessus (un candidat au-dessus du
  plafond imite un juge, il ne dépasse pas la grille) ;
- plancher : « se distingue du plancher `length` » quand l'écart apparié du chiffre principal moins `length` passe la
  règle des 2 σ, « ne se distingue pas du plancher `length` » sinon.

**La règle des 2 σ** : aucune phrase de supériorité sans que l'IC de l'écart apparié exclue zéro et dépasse 2 σ, σ =
{sigma} (le bruit des graines de s1d1, docs/18) ; lecture stricte : borne basse > 2 σ = {two_sigma} (ou, pour un écart
négatif, borne haute < −{two_sigma}). Le tableau des écarts appariés l'applique à chaque candidat ; hors de ce
tableau, seules les deux prédictions, écrites avant les chiffres, sont lues sur les valeurs ponctuelles, et le disent.

## Mises en garde (docs/32 § 2)

**Ce qui est mesuré.** Pour chaque classifieur, l'accord de ses sorties avec la médiane des juges : palier de
complexité (QWK, exactitude, exactitude à ± 1 palier, alpha ordinal, matrice de confusion), score (MAE contre l'ancre
du palier médian, diagramme de fiabilité, erreur de calibration attendue, pour les seuls scores sur l'échelle v7 ;
corrélation de rang et AUROC pour les autres scores), tâche (exactitude, F1 macro), `needs_tools` (F1, MCC, AUROC,
courbe précision-rappel, sur peu de positifs : l'IC le dit). Le palier de complexité est mesuré globalement, par langue,
par source, par origine et par palier de la vérité ; la tâche, `needs_tools` et le score le sont globalement seulement.

**Plafond et plancher.** Plafond = accord de chaque juge principal avec la médiane des autres ; un classifieur
au-dessus imite un juge, il ne dépasse pas la grille. Plancher = les témoins : aléatoire (dix tirages), longueur du
prompt (en caractères, seuils par quantiles) et `constant-low`. Entre les deux, le bruit des graines de s1d1 (docs/18)
en bande : σ ≈ 0,005 QWK, repris comme nombre, pas remesuré.

**Ce qui n'est pas mesuré.**

- (a) Si la grille elle-même correspond à « quel modèle suffit » : c'est l'autre banc, celui des réponses, laissé de
  côté par décision de l'utilisateur (2026-10-28).
- (b) Le routage (le passage du score au modèle) : affaire du déployeur.
- (c) La distribution réelle des prompts : le jeu est construit par cases, et `expert` devait être la case la moins
  fournie. Ids par palier de `truth_all` : {tier_counts} ([`tables/by_tier.csv`](tables/by_tier.csv)).
- (d) Les prompts agentiques et multi-tours : le jeu est fait de premiers prompts.
- (e) Les langues autres que les quatre choisies : docs/17 reste la seule mesure sur 78 langues.

## Comment lire

- **Les juges sont l'instrument de mesure, pas des concurrents.** Aucun tableau ne range un candidat face à un juge,
  sauf le plafond ; les juges ont leurs propres tableaux, sous ceux des candidats.
- **Les juges principaux sont dans la vérité** : {main_judges}. `truth_all` est faite de leurs votes ; leur accord avec
  elle compte donc leurs propres votes (4 sur 12 par prompt de base), et la colonne « rôle » dit « juge dans la
  vérité ». Les autres juges ({other_judges}) sont hors de la vérité : leur accord avec `truth_all` se lit avec les
  paires « même modèle » du tableau des juges, qui mesurent l'effet de la réflexion. Pour `truth_all_nogemma`, {gemma}
  sort de la vérité : avec deux juges (8 votes par prompt de base), la médiane est la moyenne des deux votes du milieu,
  arrondie vers le bas.
- **Unités** : « ids » compte les versions (une par langue, {n_ids} en tout) ; « prompts de base » compte les
  `base_key` ({n_base}). Les IC tirent les prompts de base, jamais les ids.
- **Paliers jamais prédits** : {tiers_never_predicted} ([`tables/agreement_all.csv`](tables/agreement_all.csv),
  colonne « paliers jamais prédits »). Le QWK reste sur l'échelle à cinq paliers : un candidat qui
  ne prédit jamais un palier est au mieux à un cran sur les ids de ce palier.
- **Indécidables, abstentions, votes nuls** : indécidables : {undecidable} (sortis de toute mesure de palier) ;
  abstentions : {abstentions} ; votes nuls de complexité : {null_votes} ([`tables/inputs.csv`](tables/inputs.csv), le
  tableau des dossiers ci-dessous). Un vote nul retire l'id pour ce juge seulement.
- Un nombre suivi de crochets est une valeur ponctuelle et son IC à 95 % ; « — » : non défini (par exemple un QWK sur
  un sous-groupe qui n'a qu'un palier de vérité).

## Les dossiers

{table_inputs}

Les dossiers Laya de ce rapport sont {laya_in_report}, et seulement eux. Laya répond à la question à quatre paliers de
LiteLLM et se lit nom à nom (SIMPLE → low, MEDIUM → medium, COMPLEX → high, REASONING → expert, jamais trivial), comme
l'heuristique de LiteLLM : deux candidats qui répondent à la même question se lisent avec la même table (docs/34 § 8).
Les dossiers `laya-english` et `laya-multilingual` de la PR A tiraient le palier de l'ancre la plus proche
(`nearest_anchor(attendu / 3)` : SIMPLE → trivial, MEDIUM → low, COMPLEX → high) ; ils ne sont pas dans ce rapport pour
cette raison, restent dans `classifiers/` sans modification, et `bench run` ne les charge pas. Dossiers de la racine
des classifieurs que ce run n'a pas chargés : {folders_not_in_report}.

{table_inputs_summary}

## Accord avec la vérité

### Les candidats contre `truth_all`

{figure_qwk_bars}

Barres : QWK de chaque candidat contre `truth_all` et son IC ; lignes noires : le plafond (pointillé : chaque juge
principal, trait plein : leur moyenne) ; lignes et bande grises : le plancher ; bande bleue : ± 2 σ autour de
{reference}, étroite par nature (σ = {sigma}).

{table_agreement_all_candidates}

### Le plafond : chaque juge principal contre les deux autres

{table_ceiling}

Chaque juge principal est noté contre la médiane des deux autres : 8 votes par prompt de base, pas 12. Leur moyenne est
le plafond de la phrase.

### Le plancher

{table_floor}

`length` est le témoin informé ; `constant-low` dit `low` partout ; `random` : ses dix tirages, leur minimum, maximum et
moyenne, sans IC (la bande, ce sont eux).

{litellm_next_to_constant}

{table_exact_vs_constant_low}

### Les écarts appariés

{table_paired}

Écart du QWK de {reference} moins celui de chaque autre candidat, sur les mêmes tirages des prompts de base ; « passe »
selon la règle des 2 σ (borne basse > {two_sigma}, ou borne haute < −{two_sigma}), « ne passe pas » sinon.

### Les juges contre `truth_all` (l'instrument)

{table_agreement_all_judges}

Les juges principaux sont dans la vérité : leur accord avec elle est gonflé par leurs propres votes ; le chiffre qui
les compare à la médiane des autres est le plafond. Les juges hors de la vérité ne sont pas dans `truth_all` : leur accord
avec elle se lit avec les paires « même modèle » du tableau des juges, qui mesurent l'effet de la réflexion.

### Sans Gemma 4 : `truth_all_nogemma`

{table_agreement_nogemma_candidates}

Les mêmes candidats contre la médiane des juges principaux sans {gemma} (les deux chiffres de la décision 9 de docs/32,
que cite la décision 3 de docs/34).

{table_agreement_nogemma_judges}

### Les paires de juges

{table_judge_pairs}

Chaque paire de juges sur les ids où les deux votent (le pendant de `bench agree`) ; « même modèle (réflexion) » marque
les deux réglages d'un même modèle, sans réflexion et réflexion plafonnée à 512 : l'effet de la réflexion.

## Par langue

{figure_qwk_by_language}

QWK de chaque candidat contre `truth_all` dans chaque langue.

{table_by_language_candidates}

Colonnes `truth_all` : contre la médiane toutes langues ; colonnes `truth_lang` : contre la médiane des juges
principaux sur la version elle-même (trois votes). Les deux sont publiées côte à côte.

{table_by_language_judges}

### Changer de palier d'une langue à l'autre

{table_lang_change_candidates}

Part des prompts de base dont les quatre versions ne reçoivent pas toutes le même palier, et écart moyen entre la plus
haute et la plus basse (en crans). Les juges ont le même tableau (c'est le désaccord de langue de docs/32 § 3.3) ; un
prompt de base dont une version a un vote nul sort pour ce juge et est compté.

{table_lang_change_judges}

### Les désaccords de langue de la vérité

{figure_language_disagreements}

Ids où la médiane par langue diffère de la médiane toutes langues, par langue et par source ; dernière colonne : les
prompts de base qui ont au moins une version en désaccord.

{table_truth_lang_vs_all}

## Par source et par origine

{table_by_source_candidates}

Exactitude et QWK contre `truth_all` par source du prompt ; le QWK est « — » quand le sous-groupe n'a qu'un palier de
vérité, et son IC ne garde que les tirages où le sous-groupe en a au moins deux.

{table_by_source_judges}

{table_by_origin_candidates}

Les mêmes mesures par origine (tiré du pool ou écrit à la main).

{table_by_origin_judges}

## Par palier de la vérité

{table_by_tier_candidates}

Exactitude sur les ids de chaque palier de `truth_all` et erreur signée moyenne (palier prédit moins palier de la
vérité, en crans).

{table_by_tier_judges}

{figure_confusion}

Chaque ligne est un palier de la vérité (entre parenthèses : ses ids), chaque colonne un palier prédit ; une ligne
somme à 1.

{figure_judge_spread_by_tier}

Pour chaque palier de la vérité, combien d'ids ont un écart de 0, 1, 2… crans entre le vote le plus haut et le plus bas
des juges principaux : là où l'instrument hésite.

## Tâche et `needs_tools`

Seuls les candidats qui prédisent ces sorties ont une ligne : {task_candidates}. Sans ligne, parce qu'ils ne les
prédisent pas : {no_task_candidates}.

{table_task_candidates}

{macro_f1_note}

{table_task_judges}

{table_needs_tools_candidates}

{needs_tools_note} L'AUROC n'existe que pour les candidats qui donnent une probabilité.

{table_needs_tools_judges}

{figure_needs_tools_pr}

Chaque courbe suit un candidat qui donne une probabilité de `needs_tools` : en baissant le seuil, le rappel monte et la
précision change ; la ligne pointillée grise est la part des positifs, la précision d'un tirage au hasard.

## Rangs et calibration du score

{table_rank_metrics}

Corrélation de rang (Spearman) du score avec le palier de la vérité, et AUROC du score pour « la vérité est high ou
expert ». Les scores bruts ({rank_candidates}) ne sont pas sur l'échelle v7 : seul leur ordre compte, ni MAE ni
calibration pour eux ; {rank_constant}

{table_calibration}

MAE, Spearman et erreur de calibration attendue (dix cases) du score contre l'ancre du palier de la vérité (trivial
0,01, low 0,25, medium 0,50, high 0,75, expert 1,00), pour {calibrated_candidates} seulement.

{figure_reliability_s1d1}

## Les deux prédictions, en tableau

{table_predictions}

## Ce qui manque

Trois analyses de docs/32 ne sont pas produites dans ce lot :

- (i) la part des prompts notés 0,70 à 0,80 que les juges mettent en `high` ou au-dessus (la lecture de la calibration
  de docs/32 § 2) ; le rapport publie à la place la fiabilité du score contre l'ancre du palier de la vérité (docs/34,
  décision 8) ;
- (ii) les confusions entre tâches voisines (editing / summarization / extraction, chat / information_seeking) ;
- (iii) la matrice de confusion normalisée de la tâche (docs/32 § 3.4).

Rôles, candidats et panneaux de figure absents de ce run :

{missing}

## Données brutes

Tableaux : {raw_data}. Chaque figure a son CSV dans `figures/`, sous le même nom.
