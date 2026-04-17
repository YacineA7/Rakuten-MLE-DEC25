import streamlit as st

st.set_page_config(page_title="Critiques", page_icon="🔭", layout="wide")

st.title("🔭 Critiques")
st.write(
    "Cette page permet de prendre du recul sur les résultats obtenus, les limites rencontrées et les améliorations possibles."
)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Limites du projet")
    st.markdown(
        "- Descriptions manquantes ou peu fiables\n"
        "- Titres parfois trop génériques\n"
        "- Jeu de classes déséquilibré\n"
        "- Ressources machine limitées\n"
        "- Échantillonnage partiel pour le modèle image\n"
        "- Temps limité pour l'interprétabilité approfondie"
    )

with right:
    st.subheader("Difficultés rencontrées")
    st.markdown(
        "- Charge projet répartie de manière imparfaite\n"
        "- Temps d'entraînement importants\n"
        "- Contraintes techniques sur les CNN\n"
        "- Arbitrages nécessaires entre profondeur et faisabilité"
    )

# st.subheader("Perspectives d'amélioration")
# st.markdown(
#     "- Construire un vrai modèle **multimodal texte + image**\n"
#     "- Tester des modèles de langage pré-entraînés comme **BERT**\n"
#     "- Faire du **fine-tuning** d'EfficientNetB0\n"
#     "- Réentraîner sur un plus grand volume d'images\n"
#     "- Approfondir LIME et Grad-CAM\n"
#     "- Mieux traiter le multilingue"
# )

st.success(
    "Le projet montre déjà une vraie valeur, mais il ouvre surtout la voie vers une solution multimodale plus performante et plus proche d'un usage industriel."
)
