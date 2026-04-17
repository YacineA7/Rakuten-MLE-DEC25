import streamlit as st

st.set_page_config(page_title="Conclusion", page_icon="✅", layout="wide")

st.title("✅ Conclusion du projet")
st.write(
    "Le projet démontre la faisabilité de la classification automatique de produits e-commerce à partir de données multimodales."
)

st.subheader("Ce que l'on retient")
st.markdown(
    "- Le **texte** est la modalité la plus discriminante dans le cadre étudié\n"
    "- **XGBoost** est le meilleur modèle global parmi ceux testés\n"
    "- Les **images** apportent une valeur complémentaire réelle\n"
    "- Le déséquilibre de classes peut être correctement géré avec un **F1-score pondéré**\n"
    "- La qualité des données reste un levier majeur d'amélioration"
)

st.subheader("Lien avec la problématique métier")
st.write(
    "Les résultats obtenus montrent qu'un système d'assistance au catalogage est pertinent pour une marketplace comme Rakuten. "
    "Même sans modèle multimodal complet, une pré-catégorisation basée sur le texte peut déjà améliorer la qualité du catalogue et l'expérience utilisateur."
)

st.subheader("Ouverture")
st.info(
    "La suite logique du projet est le passage à une vraie solution multimodale texte-image afin d'obtenir un système plus performant, plus robuste, plus complet et plus pertinent métier, pour un usage réel de pré-catégorisation assistée."
)
