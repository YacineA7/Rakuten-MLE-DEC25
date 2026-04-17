from pathlib import Path
import streamlit as st

st.set_page_config(page_title="EDA", page_icon="📊", layout="centered")

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.title("📊 Analyse exploratoire des données")
st.write(
    "Cette page centralise les visualisations d'exploration du projet Rakuten. "
    "Les figures sont chargées automatiquement depuis le dossier `reports/figures` du dépôt."
)

st.info(
    f"Dossier attendu pour les visualisations : `{FIGURES_DIR.as_posix()}`"
)

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

# st.markdown("## Visualisations complémentaires disponibles")
# extra_figs = [
#     "viz_txt_KNN_confusion.png",
#     "viz_txt_NB_confusion.png",
#     "viz_txt_NBOpt_confusion.png",
#     "viz_txt_SVM_confusion.png",
#     "viz_txt_SVMOpt_confusion.png",
#     "viz_txt_XGB_confusion.png",
#     "viz_img_ENB_confusion.png",
#     "viz_img_ENB_courbes.png",
#     "lime_knn_interpretation.png",
#     "lime_NB_interpretation.png",
#     "lime_SVM_interpretation.png",
#     "lime_XGB_interpretation.png",
#     "gradcam_ENB_interpretation.png",
# ]

# existing_extra = [name for name in extra_figs if (FIGURES_DIR / name).exists()]
# missing_extra = [name for name in extra_figs if not (FIGURES_DIR / name).exists()]

# if existing_extra:
#     st.success("Figures complémentaires détectées dans `reports/figures`.")
#     st.code("\n".join(existing_extra))
# else:
#     st.warning("Aucune figure complémentaire détectée pour le moment.")

# with st.expander("Afficher la liste des figures complémentaires attendues"):
#     st.code("\n".join(extra_figs))
#     if missing_extra:
#         st.caption("Fichiers encore absents ou non trouvés dans l'arborescence courante.")

st.markdown("## Lecture métier")
st.write(
    "L'EDA montre que la performance des modèles dépend autant de la richesse du texte que de la qualité du catalogage initial. "
    "Les visualisations confirment que le texte porte l'essentiel du signal discriminant, tandis que l'image fournit une information complémentaire utile pour certaines familles de produits."
)
