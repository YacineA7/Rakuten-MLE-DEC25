import streamlit as st

st.set_page_config(page_title="PoC métier", page_icon="💼", layout="wide")

st.title("💼 PoC métier")
st.write(
    "Cette page illustre comment transformer le travail de modélisation en un cas d'usage simple, crédible et utile pour une marketplace."
)

st.subheader("Cas d'usage proposé")
st.info(
    "Assistant de pré-catégorisation vendeur : au moment du dépôt d'un article, le système propose automatiquement une catégorie probable."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 1. Saisie produit")
    st.write("Le vendeur renseigne un titre, une description et éventuellement une image.")
with col2:
    st.markdown("### 2. Prédiction")
    st.write("Le modèle propose une catégorie principale et des alternatives si nécessaire.")
with col3:
    st.markdown("### 3. Action métier")
    st.write("La plateforme valide automatiquement, demande une confirmation ou route vers une revue manuelle.")

st.subheader("Règles métier illustratives")
threshold = st.slider("Seuil de confiance pour validation automatique", 0.50, 0.99, 0.85, 0.01)
review = st.slider("Seuil minimal avant revue manuelle", 0.30, 0.90, 0.60, 0.01)

st.write("Logique de décision simulée :")
if threshold > review:
    st.markdown(
        f"- Si confiance > **{threshold:.2f}** : auto-préclassement\n"
        f"- Si confiance entre **{review:.2f}** et **{threshold:.2f}** : validation assistée\n"
        f"- Si confiance < **{review:.2f}** : revue humaine obligatoire"
    )
else:
    st.error("Le seuil de validation automatique doit être supérieur au seuil de revue manuelle.")

st.subheader("Valeur métier attendue")
st.markdown(
    "- Réduction du temps de mise en ligne\n"
    "- Réduction des erreurs de catégorie\n"
    "- Meilleure qualité du catalogue\n"
    "- Meilleure expérience de recherche côté client"
)
