import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
st.set_page_config(page_title="ML Retail - Test", layout="centered")
st.title("🛍️ ML Retail - Validation sur Test Set")
st.markdown("Prédictions sur les **875 clients** que les modèles n'ont jamais vus.")

# ============================================================
# CHARGEMENT (une seule fois)
# ============================================================
@st.cache_data
def load_test_data():
    """Charge les données de test déjà prétraitées"""
    X_test = pd.read_csv('../data/train_test/X_test_pca.csv')
    X_test_scaled = pd.read_csv('../data/train_test/X_test_scaled.csv')
    y_test = pd.read_csv('../data/train_test/y_test.csv')['Churn']
    y_reg_test = pd.read_csv('../data/train_test/y_reg_test.csv')['MonetaryTotal']
    return X_test, X_test_scaled, y_test, y_reg_test

@st.cache_resource
def load_models():
    """Charge les 3 modèles entraînés"""
    churn = joblib.load('../models/churn_classifier.pkl')
    reg = joblib.load('../models/monetary_regressor.pkl')
    kmeans = joblib.load('../models/customer_clusters.pkl')
    return churn, reg, kmeans

# Chargement
try:
    X_test, X_test_scaled, y_test, y_reg_test = load_test_data()
    churn_model, reg_model, kmeans_model = load_models()
except Exception as e:
    st.error(f"❌ Erreur chargement : {e}")
    st.stop()

# ============================================================
# PRÉDICTIONS
# ============================================================
with st.spinner("Prédictions en cours..."):
    # Classification
    churn_proba = churn_model.predict_proba(X_test)[:, 1]
    churn_pred = churn_model.predict(X_test)
    
    # Régression
    monetary_pred = reg_model.predict(X_test_scaled)
    
    # Clustering
    clusters = kmeans_model.predict(X_test)

# ============================================================
# RÉSULTATS
# ============================================================
results = pd.DataFrame({
    'Client_ID': range(len(y_test)),
    'Churn_Réel': y_test.values,
    'Churn_Prédit': churn_pred,
    'Probabilité_Churn': churn_proba.round(3),
    'Montant_Réel': y_reg_test.values.round(2),
    'Montant_Prédit': monetary_pred.round(2),
    'Cluster': clusters
})

# --- Métriques ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, churn_pred)
prec = precision_score(y_test, churn_pred)
rec = recall_score(y_test, churn_pred)
f1 = f1_score(y_test, churn_pred)

st.subheader("📊 Métriques Classification")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("Precision", f"{prec:.3f}")
c3.metric("Recall", f"{rec:.3f}")
c4.metric("F1-Score", f"{f1:.3f}")

# --- Tableau ---
st.subheader("📋 Résultats détaillés")
st.dataframe(results.head(20), use_container_width=True)

# --- Téléchargement ---
csv = results.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger tout (CSV)", csv, "predictions_test.csv", "text/csv")

# ============================================================
# VISUALISATIONS
# ============================================================
st.markdown("---")
st.subheader("📈 Visualisations")

tab1, tab2, tab3 = st.tabs(["Matrice de confusion", "Régression", "Clusters"])

with tab1:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, churn_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de Confusion")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    ax.scatter(y_reg_test, monetary_pred, alpha=0.5, color='teal')
    min_max = [min(y_reg_test.min(), monetary_pred.min()), 
               max(y_reg_test.max(), monetary_pred.max())]
    ax.plot(min_max, min_max, 'r--', label='Parfait')
    ax.set_xlabel("Montant Réel (£)")
    ax.set_ylabel("Montant Prédit (£)")
    ax.set_title(f"Régression (R² = {1 - ((y_reg_test - monetary_pred)**2).sum() / ((y_reg_test - y_reg_test.mean())**2).sum():.3f})")
    ax.legend()
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], 
                        c=clusters, cmap='viridis', alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Clusters (K={len(np.unique(clusters))})")
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)

# ============================================================
# FILTRE PAR CLIENT
# ============================================================
st.markdown("---")
st.subheader("🔍 Rechercher un client spécifique")
client_id = st.number_input("Numéro du client (0 à 874)", min_value=0, max_value=874, value=0)

client = results.iloc[client_id]
col1, col2 = st.columns(2)

with col1:
    st.write("**Classification :**")
    st.write(f"- Churn réel : {'Oui' if client['Churn_Réel'] else 'Non'}")
    st.write(f"- Churn prédit : {'Oui' if client['Churn_Prédit'] else 'Non'}")
    st.write(f"- Probabilité : {client['Probabilité_Churn']*100:.1f}%")
    
    if client['Probabilité_Churn'] > 0.7:
        st.error("🔴 Risque CRITIQUE")
    elif client['Probabilité_Churn'] > 0.4:
        st.warning("🟠 Risque ÉLEVÉ")
    else:
        st.success("🟢 Risque FAIBLE")

with col2:
    st.write("**Régression :**")
    st.write(f"- Montant réel : £{client['Montant_Réel']}")
    st.write(f"- Montant prédit : £{client['Montant_Prédit']}")
    st.write(f"- Erreur : £{abs(client['Montant_Réel'] - client['Montant_Prédit']):.2f}")
    st.write(f"- Cluster : {client['Cluster']}")

st.caption("Atelier ML - GI2 | Données de test jamais vues par les modèles")