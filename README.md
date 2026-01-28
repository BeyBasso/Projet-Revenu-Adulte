
Réalisé dans le cadre du cours Application Web Interactive de Visualisation de Données. 
Auteurs : 
BASSOLE Martine Bienvenue
KOULETE Martiale

# Adult Income Predictor & Fairness Analysis

Cette application Web interactive, développée avec **Streamlit**, analyse le jeu de données "Census Income" de 1994. L'objectif est de prédire si un individu gagne plus de **50 000 $ par an** tout en évaluant l'équité (Fairness) des données et des modèles algorithmiques.


## Présentation du Projet
Ce projet s'inscrit dans le cadre de création d'applications Web basées sur la donnée. Il explore la relation entre des attributs socio-démographiques (éducation, âge, occupation) et le niveau de revenu.

### Objectifs :
1. **Exploration de données (EDA)** : Visualiser les facteurs clés de succès financier.
2. **Analyse éthique** : Détecter et mesurer les biais (notamment de genre) dans les données historiques.
3. **Modélisation** : Entraîner un modèle de Machine Learning et évaluer sa performance globale ainsi que son équité.

---

## 📂 Structure de l'application
L'application est structurée en quatre sections distinctes pour une compréhension progressive :

🏠 Accueil : Présentation du dataset (48 842 entrées), contexte sociodémographique et définition de la problématique.

📊 Exploration des Données : Visualisation des indicateurs clés (KPIs), analyse des corrélations et profilage des hauts revenus par filtres interactifs.

⚠️ Détection de Biais : Audit éthique mesurant la Parité Démographique et l'Impact Disproportionné (Ratio DI) entre les hommes et les femmes dans les données d'origine.

🤖 Modélisation & Performance : Comparaison de modèles (Régression Logistique vs Random Forest) avec évaluation des performances globales (Accuracy, Precision, Recall) et audit de fairness sur les prédictions finales.

---

📦 Fichiers du dépôt
app.py : Le code source principal de l'application Streamlit.

adult.csv : Le dataset utilisé pour l'analyse et l'entraînement.

requirements.txt : Liste des bibliothèques Python nécessaires (Pandas, Plotly, Scikit-learn, etc.).


---

📈 Résultats et Analyse de Fairness
Biais Identifié : L'analyse révèle un ratio d'impact disproportionné de ~0.30 pour les femmes dans le dataset original, bien en dessous du seuil de conformité de 0.80.

Performance du Modèle : Le modèle atteint une précision de ~82-84%. Cependant, l'audit de modélisation confirme que l'IA tend à reproduire le biais historique en prédisant moins de hauts revenus pour les femmes.

Recommandation : Ce projet démontre l'importance de ne pas se fier uniquement à l'Accuracy, mais d'auditer systématiquement l'équité des modèles de décision.

## Installation et Utilisation Locale

Pour faire tourner le projet sur votre machine :

1. **Cloner le dépôt** :
   ```bash
   git clone [https://github.com/BeyBasso/adult-income-analysis.git](https://github.com/BeyBasso/adult-income-analysis.git)
   cd adult-income-analysis