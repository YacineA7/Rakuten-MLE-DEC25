import pandas as pd
import streamlit as st

st.set_page_config(page_title="Données", page_icon="🗂️", layout="centered")

st.title("🗂️ Présentation des données")
st.write(
    "Le dataset Rakuten combine des informations textuelles et visuelles pour chaque produit. "
    "Cette double modalité structure tout le projet de classification."
)

st.subheader("Volumétrie")
summary_df = pd.DataFrame(
    {
        "Caractéristique": [
            "Volume total",
            "Train",
            "Test",
            "Nombre de classes",
            "Taille données texte",
            "Taille données images",
            "Format image",
            "Langue dominante",
        ],
        "Valeur": [
            "~99 000 produits",
            "84 916",
            "13 812",
            "27",
            "60 Mo",
            "2.2 Go",
            "JPEG RGB 500x500",
            "Français (~82% sur l'échantillon étudié)",
        ],
    }
)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Variables disponibles")
    st.markdown(
        "- `designation` : titre du produit\n"
        "- `description` : description produit\n"
        "- `productid` : identifiant produit\n"
        "- `imageid` : identifiant image\n"
        "- `prdtypecode` : catégorie cible"
    )

with col2:
    st.subheader("Architecture logique")
    st.code(
        """
X_train / X_test
 ├── métadonnées produit
 ├── texte : designation + description
 └── image : fichier JPG associé

Pipeline
 ├── préprocessing texte / image
 ├── modèles ML / DL
 └── prédiction de prdtypecode
        """
    )

st.subheader("Points structurants")
st.markdown(
    "- Le texte est riche mais hétérogène\n"
    "- La description contient de nombreuses valeurs manquantes ou peu informatives\n"
    "- Les images sont homogènes en format, ce qui simplifie leur traitement\n"
    "- Le problème est un cas de **classification multiclasse déséquilibrée**"
)
