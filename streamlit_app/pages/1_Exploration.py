import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Exploration des données", layout="wide")

st.title("Exploration des données")
st.markdown(
    """
Ce projet vise à classifier automatiquement des produits e-commerce Rakuten à partir
de deux modalités : les **textes produits** (désignation et description) et les **images produits**.
L'analyse exploratoire permet d'identifier la structure du jeu de données, les déséquilibres,
les biais potentiels et les signaux utiles pour la modélisation.
"""
)

st.info(
    "Les visualisations ci-dessous synthétisent les principaux constats issus de l'analyse exploratoire "
    "des données texte et image."
)

# Dossier courant / racine projet
ROOT = Path(__file__).resolve().parents[1]  # chemin relatif vers le dossier des visualisations

# chemins des images
img_wordcloud = ROOT / "assets/images" / "viz_txt_wordcloud.png"
img_enb_curve = ROOT / "assets/images" / "viz_img_ENB_courbes.png"
img_filesize = ROOT / "assets/images" / "viz_image_filesize.png"
img_grid = ROOT / "assets/images" / "viz_image_grid.png"
img_corr = ROOT / "assets/images" / "viz_txt_corr_len_prd.png"
img_dimensions = ROOT / "assets/images" / "viz_image_dimensions.png"
img_distrib = ROOT / "assets/images" / "viz_txt_distrib_prdtycope.png"
img_lang = ROOT / "assets/images" / "viz_txt_lang_rep.png"
img_modes = ROOT / "assets/images" / "viz_image_modes.png"

st.markdown("---")
st.header("1. Vue d’ensemble du dataset")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nombre de classes", "27")
with col2:
    st.metric("Modalités", "Texte + Image")
with col3:
    st.metric("Tâche", "Classification multiclasse")

st.markdown(
    """
L’objectif métier est d’automatiser le catalogage de produits e-commerce afin d’améliorer
la recherche, la recommandation et l’organisation du catalogue. Le projet repose sur une
approche multimodale : certaines classes sont bien identifiables par le texte, d’autres par l’image,
et certaines nécessitent la combinaison des deux.
"""
)

st.markdown("---")
st.header("2. Jeu de données")

st.markdown(
    """
Le projet repose sur un jeu de données multimodal issu du challenge Rakuten France.
Chaque produit peut être décrit à travers deux grandes sources d’information :

- **Données textuelles** : une **désignation** (titre du produit) et une **description** plus détaillée ;
- **Données visuelles** : une **image produit** associée à l’article ;
- **Variable cible** : un **code type produit** (`prdtypecode`) correspondant à la catégorie à prédire.

Le corpus global contient environ **99 000 produits**, avec des données textuelles d’environ **60 Mo**
et un volume d’images proche de **2,2 Go**. Dans notre cadre expérimental, les analyses et modèles
ont été construits sur un problème de **classification multiclasse à 27 catégories**.
"""
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Produits", "~99 000")
with col2:
    st.metric("Données texte", "~60 Mo")
with col3:
    st.metric("Données image", "~2.2 Go")
with col4:
    st.metric("Classes étudiées", "27")

st.markdown(
    """
Le dataset est **multimodal** : chaque observation combine un **titre produit**, une
**description textuelle** et une **image**. Cette structure est particulièrement adaptée
à une approche de classification combinant NLP et vision par ordinateur.
"""
)

st.subheader("Variables principales")

st.markdown(
    """
Les principales variables exploitées dans le projet sont :

- `designation` : titre court du produit ;
- `description` : texte descriptif plus riche, parfois manquant ;
- `productid` : identifiant produit ;
- `imageid` : identifiant de l’image associée ;
- `prdtypecode` : catégorie produit cible à prédire.
"""
)

st.markdown("---")
st.header("3. Exploration des données textuelles")

if img_distrib.exists():
    st.image(str(img_distrib), use_container_width=True)
    st.markdown(
        """
**Distribution des classes produits**

Cette figure met en évidence un **déséquilibre de classes important** entre les 27 `prdtypecode`.
Certaines catégories sont très représentées alors que d’autres sont nettement plus rares.
D’un point de vue métier, cela signifie qu’un modèle naïf risque d’être biaisé vers les classes
majoritaires, ce qui justifie l’usage de métriques robustes comme le **F1-score pondéré**.
"""
    )

if img_wordcloud.exists():
    st.image(str(img_wordcloud), use_container_width=True)
    st.markdown(
        """
**Nuage de mots des titres produits**

Le nuage de mots montre que certains termes sont très fréquents dans les désignations,
comme des couleurs, des tailles ou des mots génériques liés au catalogue produit.
Cette visualisation est utile pour identifier à la fois les **mots métier discriminants**
et le **bruit textuel**. Elle justifie ensuite le nettoyage, la suppression des stopwords,
et la vectorisation TF-IDF dans le pipeline Natural Language Processing (NLP).
"""
    )

col_a, col_b = st.columns(2)

with col_a:
    if img_lang.exists():
        st.image(str(img_lang), use_container_width=True)
        st.markdown(
            """
**Répartition des langues détectées**

La majorité des textes sont en **français**, avec une présence minime de l’anglais
et de quelques autres langues. Ce constat signale une certaine **hétérogénéité linguistique** dans le corpus.
Cela justifie l’utilisation de stopwords multilingues ou d’un prétraitement suffisamment robuste.
"""
        )

with col_b:
    if img_corr.exists():
        st.image(str(img_corr), use_container_width=True)
        st.markdown(
            """
**Longueur du titre selon la classe**

Cette heatmap montre que la longueur moyenne des titres varie selon les classes produits.
La longueur de texte apporte donc un **signal complémentaire**, mais elle ne suffit pas à elle seule
à discriminer correctement toutes les catégories. En pratique, cette variable peut être utile comme
feature auxiliaire, mais la sémantique du texte reste beaucoup plus informative que sa seule longueur.
Nous n'avons pas retenu la longueur comme feature principale, mais elle a été prise en compte dans l’analyse exploratoire et la compréhension du dataset.
"""
        )

st.markdown("---")
st.header("4. Exploration des données images")

col_c, col_d = st.columns(2)

with col_c:
    if img_dimensions.exists():
        st.image(str(img_dimensions), use_container_width=True)
        st.markdown(
            """
**Distribution des dimensions des images**

Les images observées sont toutes homogènes en taille, avec une taille **500 × 500 pixels**.
C’est un point positif pour la modélisation visuelle, car cela limite la variabilité technique liée
à la résolution et simplifie les étapes de redimensionnement avant l’entrée dans un réseau de neurones.
"""
        )

with col_d:
    if img_modes.exists():
        st.image(str(img_modes), use_container_width=True)
        st.markdown(
            """
**Répartition des modes colorimétriques**

Toutes les images analysées sont en **RGB**, ce qui homogénéise fortement l’entrée du pipeline image.
D’un point de vue technique, cela évite d’avoir à gérer plusieurs formats colorimétriques et réduit
les risques d’erreurs lors du prétraitement ou de la prédiction.
"""
        )

if img_filesize.exists():
    st.image(str(img_filesize), use_container_width=True)
    st.markdown(
        """
**Distribution de la taille des fichiers images**

La taille des fichiers est relativement dispersée, ce qui peut refléter des différences de compression,
de complexité visuelle ou de qualité d’image. Même si cette variable n’est pas directement utilisée comme
feature du modèle, elle donne une indication utile sur l’hétérogénéité du corpus image.
"""
    )

if img_grid.exists():
    st.image(str(img_grid), use_container_width=True)
    st.markdown(
        """
**Exemples d’images par catégorie**

Cette grille qualitative illustre la diversité visuelle des catégories. Certaines classes présentent
des objets très reconnaissables visuellement, tandis que d’autres ont des frontières plus ambiguës.
Cela confirme l’intérêt d’un modèle image de type **EfficientNetB0**, mais aussi la pertinence d’une
approche multimodale combinant image et texte.
"""
    )

st.markdown("---")
st.header("4. Premiers enseignements")

st.markdown(
    """
Les principaux constats de l’exploration sont les suivants :

- le dataset présente un **déséquilibre de classes** qui influence directement l’évaluation des modèles ;
- les **textes produits** contiennent un signal riche mais bruité, nécessitant un préprocessing complet et rigoureux ;
- les **images sont techniquement homogènes** (RGB, dimensions proches), ce qui facilite l’apprentissage ;
- certaines classes semblent mieux séparables par le texte, d’autres par l’image ;
- ces observations justifient pleinement une stratégie de **classification multimodale**.

Cette phase d’exploration a servi de base à la construction des pipelines de préprocessing,
puis à la comparaison de plusieurs modèles de Machine Learning et Deep Learning.
"""
)
