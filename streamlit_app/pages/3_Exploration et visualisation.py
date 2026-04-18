from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EDA", page_icon="📊", layout="centered")

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DICT_PATH = PROJECT_ROOT / "data" / "dictionnaire.csv"



TEXT_VIZ = [
    ("viz_txt_distrib_prdtycope.png", "Distribution des catégories", "Met en évidence le déséquilibre de classes, ce qui justifie l'usage du F1-score pondéré."),
    ("viz_txt_wordcloud.png", "Nuage de mots des désignations", "Permet d'identifier les termes fréquents, notamment les mots peu discriminants comme 'générique'."),
    ("viz_txt_lang_rep.png", "Répartition des langues", "Montre la domination du français et la présence d'anglais dans les textes produits."),
    ("viz_txt_corr_len_prd.png", "Corrélation longueur du titre / catégorie", "Aide à relier la structure textuelle des titres à certains types de produits."),
]

IMAGE_VIZ = [
    ("viz_image_dimensions.png", "Dimensions des images", "Confirme l'homogénéité du dataset image, avec un format uniforme facilitant le preprocessing."),
    ("viz_image_modes.png", "Modes colorimétriques des images", "Vérifie la cohérence des canaux et limite les besoins de conversion complexes."),
    ("viz_image_filesize.png", "Distribution de la taille des fichiers", "Donne une vue d'ensemble sur le poids des images et la variabilité de compression."),
    ("viz_image_grid.png", "Grille d'images échantillon", "Permet une inspection qualitative rapide de la diversité visuelle des produits."),
]


def show_figure(path: Path, title: str, comment: str):
    st.markdown(f"### {title}")
    if path.exists():
        st.image(str(path), use_container_width=True)
        st.caption(comment)
    else:
        st.warning(f"Figure introuvable : `{path.name}`")
        st.caption(comment)


st.markdown("## Synthèse des constats")
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown(
        "- Le dataset texte présente un **déséquilibre de classes** marqué\n"
        "- Les désignations sont souvent plus fiables que les descriptions\n"
        "- La langue dominante est le français, avec une part non négligeable d'anglais\n"
        "- Certains mots fréquents sont peu discriminants et doivent être traités avec précaution"
    )
with col2:
    st.markdown(
        "- Les images sont homogènes en taille et en format\n"
        "- La qualité globale est compatible avec une approche CNN\n"
        "- La variabilité visuelle reste forte selon les catégories\n"
        "- L'image constitue une modalité complémentaire utile au texte"
    )

st.markdown("## Visualisations texte")
for filename, title, comment in TEXT_VIZ:
    show_figure(FIGURES_DIR / filename, title, comment)

st.markdown("## Visualisations image")
for filename, title, comment in IMAGE_VIZ:
    show_figure(FIGURES_DIR / filename, title, comment)


if DICT_PATH.exists():
    df_dict = pd.read_csv(DICT_PATH)

    # Harmonisation minimale des noms de colonnes
    df_dict.columns = [c.strip().lower() for c in df_dict.columns]

    # Exemple attendu : prdtypecode + designation/libelle/categorie
    possible_label_cols = ["categorie", "libelle", "label", "designation", "type"]

    label_col = next((c for c in possible_label_cols if c in df_dict.columns), None)

    if "prdtypecode" in df_dict.columns and label_col is not None:
        df_affichage = (
            df_dict[["prdtypecode", label_col]]
            .drop_duplicates()
            .sort_values("prdtypecode")
            .rename(columns={label_col: "catégorie"})
        )

        with st.expander("Voir le dictionnaire des catégories"):
            st.write(df_dict)
    else:
        st.warning(
            "Le fichier dictionnaire.csv a été trouvé, mais il doit contenir au moins "
            "une colonne 'prdtypecode' et une colonne de libellé "
            "(ex. categorie, libelle, label, designation)."
        )
else:
    st.info("Fichier introuvable : ajoute 'data/dictionnaire.csv' dans le projet.")

st.markdown("## Lecture métier")
st.write(
    "L'exploration montre que la performance des modèles dépend autant de la richesse du texte que de la qualité du catalogage initial. "
    "Les visualisations confirment que le texte porte l'essentiel du signal discriminant, tandis que l'image fournit une information complémentaire utile pour certaines familles de produits."
)