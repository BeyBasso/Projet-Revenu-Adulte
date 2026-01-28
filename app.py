import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# =========================================================================
# 0. FONCTIONS UTILITAIRES (Fairness & Data)
# =========================================================================

def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    """Calcule la différence de parité démographique."""
    df_temp = pd.DataFrame({'y': y_pred, 'group': sensitive_attribute})
    rates = df_temp.groupby('group')['y'].mean()
    # Différence absolue entre les taux des deux groupes
    return {'difference': abs(rates.iloc[0] - rates.iloc[1])}

def disparate_impact_ratio(y_true, y_pred, sensitive_attribute, unprivileged_value, privileged_value):
    """Calcule le ratio d'impact disproportionné."""
    df_temp = pd.DataFrame({'y': y_pred, 'group': sensitive_attribute})
    prob_unprivileged = df_temp[df_temp['group'] == unprivileged_value]['y'].mean()
    prob_privileged = df_temp[df_temp['group'] == privileged_value]['y'].mean()
    ratio = prob_unprivileged / prob_privileged if prob_privileged > 0 else 0
    return {'ratio': ratio}

@st.cache_data
def load_data():
    if not os.path.exists('adult.csv'):
        st.error("Erreur : Le fichier 'adult.csv' est introuvable.")
        st.stop()
    data = pd.read_csv('adult.csv')
    # Nettoyage des espaces
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    # Création de la variable cible numérique (1 si >50K, 0 sinon)
    data['target'] = (data['income'] == '>50K').astype(int)
    return data

# Configuration Streamlit
st.set_page_config(page_title="Adult Income Analysis", page_icon="💰", layout="wide")
df = load_data()

# Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Sélectionnez une page :", 
    ["🏠 Accueil", "📊 Exploration des Données", "⚠️ Détection de Biais", "🤖 Modélisation"])

# =========================================================================
# PAGE 1 : 🏠 ACCUEIL
# =========================================================================
if page == "🏠 Accueil":
    st.title("🏠 Accueil - Projet Adult Income")
    
    st.header("Présentation du Dataset")
    st.markdown("""
    Le projet s'appuie sur le jeu de données **"Adult Census Income"**, extrait de la base de données du Bureau du recensement des États-Unis (1994). 
    Il contient environ **48 842 entrées** et **15 variables**. Chaque ligne représente un individu avec des attributs socio-démographiques tels que l'âge, l'éducation, 
    le statut marital, l'occupation, et le temps de travail hebdomadaire.
    """)

    st.header("Contexte et Problématique")
    st.markdown("""
    Dans une économie moderne, comprendre les facteurs qui influencent la prospérité financière individuelle est un enjeu majeur pour les politiques publiques. 
    Ce dataset permet d'étudier comment les variables méritocratiques (comme l'éducation) et les variables structurelles (comme le genre ou la race) 
    interagissent pour définir le niveau de vie d'un citoyen.

    La problématique centrale est de savoir s'il est possible de **prédire avec précision si un individu perçoit un revenu annuel supérieur à 50 000 $** en se basant sur son profil. Au-delà de la performance, nous cherchons à identifier les **biais historiques** (genre, race) afin d'éviter 
    que les futurs modèles automatisés ne reproduisent ces inégalités.
    """)

# =========================================================================
# PAGE 2 : 📊 EXPLORATION DES DONNÉES
# =========================================================================
elif page == "📊 Exploration des Données":
    st.title("📊 Exploration des Données")
    
    # Bonus: Filtres interactifs
    st.sidebar.header("🔍 Filtres")
    selected_race = st.sidebar.multiselect("Race", options=df['race'].unique(), default=df['race'].unique())
    selected_gender = st.sidebar.multiselect("Genre", options=df['gender'].unique(), default=df['gender'].unique())
    df_filtered = df[(df['race'].isin(selected_race)) & (df['gender'].isin(selected_gender))]

    st.header("Analyse Globale")
    
    # 4 KPIs Obligatoires
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Lignes", len(df_filtered))
    with k2:
        st.metric("Total Colonnes", df_filtered.shape[1])
    with k3:
        missing_rate = (df_filtered == '?').sum().sum() / df_filtered.size * 100
        st.metric("Taux de '?'", f"{missing_rate:.2f}%")
    with k4:
        st_rate = (df_filtered['target'] == 1).mean() * 100
        st.metric("Taux >50K", f"{st_rate:.1f}%")

    # Focus >50K
    st.markdown("#### 💎 Profil des hauts revenus (>50K)")
    df_high = df_filtered[df_filtered['target'] == 1]
    f1, f2 = st.columns(2)
    f1.metric("Âge moyen", f"{df_high['age'].mean():.1f} ans" if not df_high.empty else "N/A")
    f2.metric("Travail hebdomadaire moyen", f"{df_high['hours-per-week'].mean():.1f} h" if not df_high.empty else "N/A")

    st.divider()

    # Visualisations
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("1. Distribution de la variable cible")
        st.plotly_chart(px.histogram(df_filtered, x="income", color="income"), use_container_width=True)
    
    with col_v2:
        st.subheader("2. Comparaison des revenus par Genre")
        st.plotly_chart(px.histogram(df_filtered, x="gender", color="income", barmode="group"), use_container_width=True)

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        st.subheader("3. Heatmap des corrélations")
        corr = df_filtered.select_dtypes(include=[np.number]).corr()
        st.plotly_chart(go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu')), use_container_width=True)
    
    with col_v4:
        st.subheader("4. Proportions par Lien de parenté")
        st.plotly_chart(px.pie(df_filtered, names='relationship', hole=0.3), use_container_width=True)

    with st.expander("📄 Aperçu des données"):
        st.dataframe(df_filtered.head(50))

# =========================================================================
# PAGE 3 : ⚠️ DÉTECTION DE BIAIS
# =========================================================================
elif page == "⚠️ Détection de Biais":
    st.title("⚠️ Détection de Biais")

    # Visualisation 1 : Distribution cible
    st.subheader("Visualisation 1 : Distribution de la variable cible")
    fig1 = px.histogram(df, x="target", title="Distribution de la cible (0: <=50K, 1: >50K)")
    st.plotly_chart(fig1, use_container_width=True)

    # Visualisation 2 : Comparaison par groupe
    st.subheader("Visualisation 2 : Comparaison par groupe (Genre)")
    df_grouped = df.groupby(['gender', 'target']).size().reset_index(name='count')
    fig2 = px.bar(df_grouped, x='gender', y='count', color='target', barmode="group", title="Répartition par Genre")
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # 1. Explication
    st.header("1. Explication du biais analysé")
    st.markdown("""
    - **Attribut sensible** : Genre (`gender`).
    - **Pourquoi c'est problématique ?** : Les inégalités salariales historiques sont présentes dans les données. Si un modèle apprend que les femmes 
    gagnent moins en moyenne, il risque de pénaliser systématiquement les femmes dans des processus automatisés de recrutement ou de crédit.
    """)

    # 2. Métriques de Fairness
    st.header("2. Métriques de Fairness")
    col_m1, col_m2 = st.columns(2)
    
    res_dp = demographic_parity_difference(df['target'].values, df['target'].values, df['gender'].values)
    with col_m1:
        st.metric("Différence de Parité Démographique", f"{res_dp['difference']:.2f}")

    res_di = disparate_impact_ratio(df['target'].values, df['target'].values, df['gender'].values, 'Female', 'Male')
    with col_m2:
        st.metric("Ratio d'Impact Disproportionné (DI)", f"{res_di['ratio']:.2f}")

    # 3. Visualisation des résultats
    st.subheader("3. Taux de hauts revenus par Genre")
    rates = df.groupby('gender')['target'].mean().reset_index()
    st.plotly_chart(px.bar(rates, x='gender', y='target', title="Taux de succès par groupe"), use_container_width=True)

    # 4. Interprétation
    st.header("4. Interprétation")
    st.markdown(f"""
    Le biais détecté est significatif : le ratio DI de **{res_di['ratio']:.2f}** est bien inférieur au seuil de **0.80**. 
    Cela signifie que le groupe **Femme** est défavorisé. L'impact réel serait une discrimination systémique 
    réduisant les opportunités financières des femmes. Il est recommandé de rééquilibrer le dataset avant l'entraînement.
    """)

# =========================================================================
# PAGE 4 : 🤖 MODÉLISATION
# =========================================================================
elif page == "🤖 Modélisation":
    st.title("🤖 Modélisation et Évaluation")
    
    with st.spinner("Entraînement du modèle..."):
        # Pre-processing simple
        feat = ['age', 'educational-num', 'hours-per-week', 'gender']
        X = pd.get_dummies(df[feat], drop_first=True)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)
        
        clf = LogisticRegression().fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

    # Performances Globales
    st.header("1. Performances Globales")
    c_p1, c_p2, c_p3 = st.columns(3)
    c_p1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    c_p2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    c_p3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")

    # Fairness sur prédictions
    st.header("2. Fairness sur les Prédictions")
    test_df = X_test.copy()
    test_df['y_pred'] = y_pred
    rate_m = test_df[test_df['gender_Male'] == 1]['y_pred'].mean()
    rate_f = test_df[test_df['gender_Male'] == 0]['y_pred'].mean()
    st.metric("Ratio DI (Prédictions)", f"{(rate_f/rate_m):.2f}")

    # Matrices de Confusion par Genre
    st.header("3. Matrices de Confusion par Groupe")
    col_cm1, col_cm2 = st.columns(2)
    for g, col, lab in [(1, col_cm1, "Homme"), (0, col_cm2, "Femme")]:
        subset = test_df[test_df['gender_Male'] == g]
        cm = confusion_matrix(y_test[X_test['gender_Male'] == g], subset['y_pred'])
        col.write(f"**Matrice : {lab}**")
        col.plotly_chart(go.Figure(data=go.Heatmap(z=cm, x=['P <=50K', 'P >50K'], y=['V <=50K', 'V >50K'], colorscale='YlGnBu')), use_container_width=True)

    st.info("Le modèle préserve le biais : il prédit beaucoup moins souvent un haut revenu pour les femmes.")