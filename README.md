# 🏟️ Arène des Algos — Ryma Dinari

> Pipeline ML complet de bout en bout : exploration, nettoyage, modélisation,
> évaluation rigoureuse et déploiement. Construit sur 4 jours, dataset par dataset.


---

## 📅 Jour 1 — Le pipeline supervisé de bout en bout

### Dataset
**Breast Cancer** (sklearn) — 569 patients, 30 mesures, 2 classes (maligne/bénigne)
**Wine** (sklearn) — 178 vins, 13 mesures, 3 classes

### Ce qu'on a fait
- Exploration du dataset : répartition des classes, détection déséquilibre
- Pipeline complet : split → entraînement → prédiction → mesure
- Arène : 3 algos sur le même split, classement trié par accuracy
- Clustering KMeans sans étiquettes : retrouve les vraies classes ?
- Visualisations : barplot accuracies + matrice de confusion
- Scaling + démonstration data leakage chiffres en main

### Arène Breast Cancer — avec scaling

| Rang | Algorithme | Sans scaling | Avec scaling | Gain |
|------|-----------|-------------|-------------|------|
| 🥇 | Régression logistique | 94.7% | 98.2% | +3.5 pp |
| 🥈 | KNN | 95.6% | 97.4% | +1.8 pp |
| 🥉 | Arbre de décision | 93.9% | 93.9% | +0.0 pp |

### Arène Wine — sans scaling

| Rang | Algorithme | Accuracy |
|------|-----------|---------|
| 🥇 | Arbre de décision | 94.4% |
| 🥈 | KNN | 91.7% |
| 🥉 | Régression logistique | 86.1% |

### Leçons clés
- Le champion change selon le dataset
- Le scaling est décisif pour KNN et régression logistique, neutre pour les arbres
- Data leakage : `scaler.fit()` uniquement sur `X_train`, jamais sur tout X
- KMeans retrouve les vraies classes sans jamais voir les étiquettes → structure dans les données

---

## 📅 Jour 2 — Nettoyage de données réelles (Telco Churn)

### Dataset
**Telco Customer Churn** — 7043 clients, 21 colonnes
Cible : `Churn` (le client a-t-il résilié ?)

### Ce qu'on a fait
- Audit qualité : détection déséquilibre 73/27, trous cachés
- Réparation `TotalCharges` : 11 espaces cachés → NaN → imputation médiane
- Encodage : Yes/No → 0/1, One-Hot pour les nominales, suppression customerID
- Détection outliers IQR + boxplots + décision justifiée par colonne
- Multicolinéarité VIF : suppression `TotalCharges` (≈ tenure × MonthlyCharges)
- Features discriminantes : info mutuelle + Random Forest
- Split stratifié + scaling anti-leakage
- `DataCleaner` réutilisable (fit/transform) + version industrielle sklearn Pipeline
- Crash-test sur nouveau dataset

### Tableau de bord du nettoyage

| Étape | Avant | Après |
|-------|-------|-------|
| Colonnes | 21 | 31 (après One-Hot) |
| Lignes | 7043 | 7043 (imputation, aucune perdue) |
| Trous cachés | 11 | 0 |
| Colonnes VIF > 5 | 2 | 0 |
| customerID | présent | supprimé |

### Top features prédictives du Churn

| Rang | Feature | Pourquoi |
|------|---------|---------|
| 1 | Contract (month-to-month) | Sans engagement = plus de départs |
| 2 | tenure | Client récent = risque élevé |
| 3 | MonthlyCharges | Facture élevée = insatisfaction |

### Choix justifiés
**Imputer vs supprimer** : 11 lignes sur 7043 = 0.15% → imputation médiane, aucune perte.

**Contract : nominal ou ordinal ?** One-Hot choisi. L'effet sur le churn n'est pas
linéaire — un encodage ordinal 0/1/2 supposerait une progression régulière non prouvée.

**Outliers conservés** : une facture élevée = client premium réel, pas une erreur de saisie.

### Règle d'or DataCleaner
```python
cleaner = DataCleaner()
X_train_clean = cleaner.fit_transform(X_train)  # apprend sur le train
X_test_clean  = cleaner.transform(X_test)        # rejoue sans recalculer
```

---

## 📅 Jour 3 — L'Arène sur 4 problèmes réels

### Phase A — Régression (California Housing)

| Modèle | R2 | MAE | RMSE |
|--------|-----|-----|------|
| Régression linéaire | 0.58 | 0.53 | 0.75 |
| Random Forest | 0.80 | 0.33 | 0.51 |

> Le Random Forest écrase la linéaire : les relations prix/variables ne sont pas linéaires.

### Phase B — Clustering AirBnB (non supervisé)

- KMeans k=3 retenu (meilleure silhouette)
- Segments : "Petits budgets" / "Premium" / "Familial longue durée"
- Sans standardiser : la colonne prix écrase tout → clusters sans sens
- 1 outlier à 100k€ déforme tous les centres de clusters

### Phase C — Spam SMS (texte)

| Modèle | Precision spam | Recall spam | F1 spam |
|--------|---------------|------------|---------|
| Naive Bayes | 0.97 | 0.91 | 0.94 |
| Régression logistique | 0.99 | 0.93 | 0.96 |

> Sur le spam : un faux positif (vrai mail en spam) est souvent pire qu'un faux négatif.
> L'accuracy seule ment sur données déséquilibrées.

### Phase D — Sonar mines/rochers

| Modèle | Avec scaling | Sans scaling |
|--------|-------------|-------------|
| SVC rbf | 0.86 | 0.62 |
| Régression logistique | 0.76 | 0.64 |
| Random Forest | 0.83 | 0.81 |

> SVM = terrain idéal sur peu de données + beaucoup de variables.
> Random Forest insensible au scaling (raisonne par seuils, pas distances).

### Phase E — Fight des IA (Leaderboard Sonar, F1)

| Rang | Algorithme | F1 | Temps |
|------|-----------|-----|-------|
| 🥇 | SVC_rbf | 0.87 | 0.01s |
| 🥈 | RandomForest | 0.84 | 0.31s |
| 🥉 | GradientBoosting | 0.82 | 0.42s |
| 4 | LogisticRegression | 0.78 | 0.02s |
| 5 | DecisionTree | 0.71 | 0.01s |

---

## 📅 Jour 4 — Évaluation rigoureuse + Déploiement

### Ce qu'on a fait
- Split propre train / validation / test (3 jeux, 3 rôles)
- Bootstrap : mesure de stabilité par rééchantillonnage avec remise
- Validation croisée k-fold : moyenne et écart-type fiables
- Métriques métier : coût des faux négatifs vs faux positifs
- Sérialisation joblib + API Flask de prédiction
- WebApp Streamlit interactive avec alertes hors plage

### Pourquoi 3 jeux de données ?

| Jeu | Rôle |
|-----|------|
| Train (60%) | Le modèle apprend |
| Validation (20%) | On règle les choix d'algo |
| Test (20%) | Verdict final — touché une seule fois |

### Leaderboard final — Breast Cancer

| Modèle | CV Accuracy | Recall | Coût métier | Temps |
|--------|------------|--------|------------|-------|
| 🥇 RandomForest | 0.963 | 0.97 | 48 | 0.31s |
| 🥈 GradientBoosting | 0.958 | 0.96 | 56 | 0.42s |
| 🥉 SVC_rbf | 0.955 | 0.95 | 62 | 0.01s |
| LogisticRegression | 0.953 | 0.94 | 71 | 0.02s |
| DecisionTree | 0.921 | 0.91 | 98 | 0.01s |

### 🏆 Champion final : Random Forest

| Critère | Détail |
|---------|--------|
| 🎯 CV Accuracy | 0.963 — stable sur 5 folds |
| 🏥 Recall | 0.97 — rate très peu de tumeurs malignes |
| 💰 Coût métier | 48 — le plus bas du leaderboard |
| ⚡ Vitesse | 0.31s — acceptable en production |

**Pourquoi pas le réseau de neurones ?**
Sur données tabulaires structurées, un bon algo classique bat souvent
un PMC tout en étant 10x plus rapide et plus interprétable.

### API — comment l'utiliser
```bash
# Lancer l'API
python api.py

# Envoyer une prédiction
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [17.99, 10.38, 122.8, ...]}'

# Réponse
{"prediction": 0, "proba": 0.97, "label": "maligne"}
```

### WebApp
```bash
streamlit run app.py
# Ouvre automatiquement http://localhost:8501
```

---

## 💡 Les 10 réflexes retenus cette semaine

1. **Split avant tout** — jamais de fit sur le test, jamais
2. **Scaler sur le train seul** — `fit_transform(X_train)`, `transform(X_test)`
3. **Stratify sur données déséquilibrées** — conserver les proportions dans chaque jeu
4. **L'accuracy ment** — sur du 73/27, regarder recall et coût métier
5. **Standardiser avant KMeans et SVM** — sans ça les distances n'ont aucun sens
6. **Un champion dépend du dataset** — tester sur plusieurs avant de conclure
7. **Commitez souvent** — un commit = une phase, message clair
8. **fit/transform séparés** — `DataCleaner.fit()` apprend, `.transform()` rejoue
9. **Outlier ≠ erreur** — un client à facture élevée est un cas réel précieux
10. **Coût métier > accuracy** — rater une tumeur maligne coûte bien plus qu'un faux positif

---

## 🛠️ Stack technique

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![pandas](https://img.shields.io/badge/pandas-2.0-purple)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3-red)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7-green)
