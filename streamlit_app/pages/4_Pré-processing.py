import pandas as pd
import streamlit as st

st.set_page_config(page_title="Préprocessing", page_icon="🧹", layout="centered")

st.title("🧹 Préprocessing et justification")
st.write(
    "Le préprocessing vise à transformer les données brutes en entrées exploitables, plus propres et plus cohérentes pour les modèles."
)

tab1, tab2, tab3 = st.tabs(["Texte", "Image", "Justification"])

with tab1:
    st.subheader("Pipeline texte")
    text_df = pd.DataFrame(
        {
            "Étapes": [
                "Suppression HTML",
                "Suppression URLs",
                "Normalisation espaces",
                "Passage en minuscules",
                "Suppression ponctuation",
                "Concaténation designation + description",
                "Suppression stopwords FR/EN",
                "Suppression du terme 'générique'",
                "Stemming français avec Snowball ",
                "Vectorisation TF-IDF",
                "Encodage LabelEncoder",
            ]
        }
    )
    st.dataframe(text_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Pipeline image")
    image_df = pd.DataFrame(
        {
            "Étapes": [
                "Lecture des métadonnées",
                "Reconstruction du chemin image",
                "Redimensionnement en 224x224",
                "Conversion BGR vers RGB",
                "Normalisation float32",
                "Preprocess spécifique EfficientNetB0",
                "Vérification d'existence des fichiers",
                "Fallback image noire si image manquante/illisible",
            ]
        }
    )
    st.dataframe(image_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Pourquoi ces choix ?")
    st.markdown(
        "- Le nettoyage textuel réduit le bruit textuel\n"
        "- Le stemming rapproche les variantes lexicales\n"
        "- TF-IDF est adapté aux modèles linéaires et de boosting sur texte\n"
        "- EfficientNetB0 impose un format image standardisé\n"
        "- Le fallback image noire rend l'inférence plus robuste"
    )
    st.success(
        "Le préprocessing est directement guidé par les constats de l'exploration : bruit textuel, descriptions incomplètes, homogénéité des images et besoin de robustesse."
    )
