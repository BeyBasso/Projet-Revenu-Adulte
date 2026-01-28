# Adult Income Predictor & Fairness Audit

## 📊 Description

Ce projet s'inscrit dans une démarche d'audit algorithmique appliquée aux données socio-économiques. Le contexte repose sur l'exploitation du jeu de données "Census Income" de 1994, une base historique permettant d'étudier les facteurs influençant le niveau de richesse aux États-Unis.

L'objectif principal est de concevoir un outil capable de prédire si un individu perçoit un revenu annuel supérieur à 50 000 $, tout en identifiant de manière critique les biais discriminatoires (notamment de genre) présents dans les données d'entraînement.

L'application permet aux utilisateurs d'explorer visuellement le dataset, de mesurer statistiquement les disparités de traitement entre les groupes (hommes/femmes) et de tester la performance de modèles de Machine Learning tout en auditant leur équité décisionnelle.

## 🎯 Parcours

- **Parcours A** : Détection de Biais

## 📁 Dataset

- Source : Dataset "UCI Adult Income" (Census Income 1994).

- Taille : 48 842 lignes, 15 colonnes.

- Variables principales : age, educational-num (années d'études), gender, race, hours-per-week, occupation.

- Variable cible : income (binaire : <=50K ou >50K).

## 🚀 Fonctionnalités

### Page 1 : Accueil
- Présentation détaillée du dataset et de la problématique.

- Explication du contexte et des enjeux éthiques de l'IA.

### Page 2 : Exploration
- Affichage de 4 KPIs : Total lignes, colonnes, taux de valeurs manquantes et taux de hauts revenus.

- Filtres interactifs par Race et Genre.

- Visualisations : Distributions cibles, heatmap de corrélation, boxplot des âges et pie chart des relations.

### Page 3 : [Détection de Biais / Analyse Approfondie]
- Calcul des métriques de Fairness : Demographic Parity Difference et Disparate Impact Ratio.

- Visualisation des taux de succès comparés entre les genres.

- Section d'interprétation des résultats de biais.

### Page 4 (Bonus) : [Si applicable]
- Entraînement de modèles (Logistic Regression / Random Forest).

- Évaluation des performances (Accuracy, Precision, Recall).

- Audit de fairness sur les prédictions via des matrices de confusion séparées par sexe.

## 🛠️ Technologies Utilisées

- Python 3.x
- Streamlit
- Pandas
- Plotly Express


## 📦 Installation Locale
```bash
# Cloner le repository
git clone https://github.com/BeyBasso/Projet-Revenu-Adulte
cd Projet-Revenu-Adulte

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## 🌐 Déploiement

Application déployée sur Streamlit Cloud :
👉 [Lien vers l'application](https://projet-revenu-adulte-6h2pgtbqhx4bqoef7xkf2v.streamlit.app/)

## 👥 Équipe

- **[BASSOLE Martine Bienvenue]** - [code app.py et redaction du fichier readm.md]
- **[KOULETE Martiale]** - [Deploiement sur strealit]


## 📝 Notes

[Optionnel : Difficultés rencontrées, améliorations futures, etc.]