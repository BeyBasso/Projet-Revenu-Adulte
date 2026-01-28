# Adult Income Predictor & Fairness Analysis

Cette application Web interactive, développée avec **Streamlit**, analyse le jeu de données "Census Income" de 1994. L'objectif est de prédire si un individu gagne plus de **50 000 $ par an** tout en évaluant l'équité (Fairness) des données et des modèles algorithmiques.

## Lien de l'application
[👉 Cliquez ici pour accéder à l'application en ligne](VOTRE_LIEN_STREAMLIT_ICI)

---

## Présentation du Projet
Ce projet s'inscrit dans le cadre de création d'applications Web basées sur la donnée. Il explore la relation entre des attributs socio-démographiques (éducation, âge, occupation) et le niveau de revenu.

### Objectifs :
1. **Exploration de données (EDA)** : Visualiser les facteurs clés de succès financier.
2. **Analyse éthique** : Détecter et mesurer les biais (notamment de genre) dans les données historiques.
3. **Modélisation** : Entraîner un modèle de Machine Learning et évaluer sa performance globale ainsi que son équité.

---

## 📂 Structure de l'application
L'application est divisée en 4 sections principales :

1. **🏠 Accueil** : Présentation du dataset UCI Adult et de la problématique.
2. **📊 Exploration** : KPIs globaux, indicateurs sur les hauts revenus et corrélations entre variables.
3. **⚠️ Détection de Biais** : Analyse approfondie des disparités hommes/femmes via les métriques de *Demographic Parity* et *Disparate Impact*.
4. **Modélisation** : Entraînement d'une Régression Logistique avec affichage des performances (Accuracy, Precision, Recall) et des matrices de confusion par groupe.

---

## Installation et Utilisation Locale

Pour faire tourner le projet sur votre machine :

1. **Cloner le dépôt** :
   ```bash
   git clone [https://github.com/VOTRE_PSEUDO/adult-income-analysis.git](https://github.com/VOTRE_PSEUDO/adult-income-analysis.git)
   cd adult-income-analysis