# Arène des Algos — Ryma Dinari

Projet ML de la semaine : pipeline complet de classification + comparaison d'algorithmes.

## Le problème
Comparer plusieurs algorithmes de classification ML sur deux datasets
(breast_cancer et wine) et choisir le meilleur en justifiant le choix.

## Datasets
- **Breast Cancer** : 569 patients, 30 mesures, 2 classes (maligne / bénigne)
- **Wine** : 178 vins, 13 mesures, 3 classes

## Classement final — Breast Cancer (avec scaling)

| Rang | Algorithme            | Accuracy |
|------|-----------------------|----------|
| 🥇   | Régression logistique | 98.2%    |
| 🥈   | KNN                   | 97.4%    |
| 🥉   | Arbre de décision     | 93.9%    |

## Classement final — Wine (sans scaling)

| Rang | Algorithme            | Accuracy |
|------|-----------------------|----------|
| 🥇   | Arbre de décision     | 94.4%    |
| 🥈   | KNN                   | 91.7%    |
| 🥉   | Régression logistique | 86.1%    |

## Champion retenu : Régression logistique (breast_cancer + scaling)

**Pourquoi ce choix ?**
- Meilleure accuracy (98.2%) sur le dataset médical le plus important
- Interprétable : on peut expliquer chaque décision au médecin
- Rapide à entraîner et à exécuter en production
- Ses erreurs sont limitées : très peu de faux négatifs (tumeurs malignes ratées)

**Limite principale :** nécessite un scaling des données. Sans StandardScaler,
son accuracy chute à 94.7%. Le scaler doit toujours être fitté sur le train seul.

## Structure du repo
- `notebook.ipynb` : pipeline complet (exploration, arène, clustering, scaling)
- `arene_breast_cancer.png` : barplot des accuracies
- `confusion_matrix.png` : matrice de confusion du champion

## Ce que j'ai appris
- Un algo qui gagne sur un dataset peut perdre sur un autre
- Le scaling change tout pour KNN et la régression logistique, rien pour les arbres
- Le data leakage peut gonfler artificiellement les résultats
- Un modèle interprétable vaut parfois mieux qu'un modèle légèrement plus précis
"""
