import streamlit as st

st.set_page_config(page_title="Pré-processing", layout="wide")

st.title("Pré-processing des données")

st.markdown(
    """
Cette section présente les transformations appliquées aux données **textuelles** et **images**
afin de construire des jeux de données exploitables par les modèles de Machine Learning et de Deep Learning.
L’objectif du pré-processing est double : **améliorer la qualité du signal** et **standardiser les entrées**
pour rendre les modèles plus robustes.
"""
)

st.markdown("---")
st.header("1. Objectifs du pré-processing")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Modalités traitées", "Texte + Image")
with col2:
    st.metric("Classes cibles", "27")
with col3:
    st.metric("Objectif", "Standardiser les entrées")

st.markdown(
    """
Le pré-processing a été pensé comme une étape de préparation essentielle avant la modélisation.
Sur le texte, il vise à supprimer le bruit, homogénéiser les formulations et extraire une représentation
vectorielle exploitable. Sur les images, il permet d’assurer un format cohérent, compatible avec
le modèle **EfficientNetB0** utilisé ensuite pour la classification visuelle.
"""
)

st.markdown("---")
st.header("2. Pré-processing des données textuelles")

st.subheader("Pourquoi nettoyer les textes ?")

st.markdown(
    """
Les données textuelles du projet proviennent des colonnes **`designation`** et **`description`**.
Le rapport final montre que la **désignation** constitue la source textuelle la plus fiable,
car la colonne **description** contient une part importante de valeurs manquantes ou vides. [file:101]

Le nettoyage textuel vise donc à :
- supprimer les éléments non informatifs,
- homogénéiser l’écriture,
- rapprocher les variantes lexicales d’un même mot,
- produire une représentation adaptée aux modèles de classification.
"""
)

st.subheader("Étapes appliquées sur le texte")

text_steps = [
    "Suppression des balises HTML présentes dans certaines descriptions",
    "Remplacement des références de caractères HTML (`&amp;`, `&nbsp;`, etc.)",
    "Suppression des URLs et des liens",
    "Normalisation des espaces et nettoyage des caractères parasites",
    "Passage de tous les textes en minuscules",
    "Suppression de la ponctuation",
    "Concaténation de `designation` et `description` après nettoyage",
    "Suppression des variables `productid` et `imageid` du pipeline textuel",
    "Suppression des stopwords français et anglais avec NLTK",
    "Ajout du mot `générique` comme stopword spécifique métier",
    "Suppression des tokens trop courts",
    "Application d’un stemming français avec `SnowballStemmer`",
    "Comptage des doublons dans les textes nettoyés",
    "Vectorisation finale avec `TfidfVectorizer`",
    "Encodage de la cible `prdtypecode` avec `LabelEncoder` pour obtenir 27 labels"
]

for step in text_steps:
    st.markdown(f"- {step}")

st.subheader("Choix méthodologiques")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
**Concaténation des champs texte**

La concaténation de `designation` et `description` permet de conserver à la fois
l’information courte et très discriminante du titre, ainsi qu’une information plus détaillée
quand la description est disponible. Cela permet de maximiser le signal textuel utile
avant vectorisation.
"""
    )

with col_b:
    st.markdown(
        """
**Stemming et stopwords**

Le rapport indique que le français est la langue dominante du corpus, avec une présence plus faible
de l’anglais. Le choix d’un **stemmer français** a donc été retenu pour simplifier le pipeline,
tout en capturant une partie des variations lexicales. Les stopwords anglais et français ont été retirés,
ainsi que le mot **“générique”**, jugé peu discriminant.
"""
    )

st.subheader("Représentation finale du texte")

st.code(
    """
# Schéma logique du pipeline texte
designation + description
    -> nettoyage HTML / URLs / ponctuation
    -> minuscules / normalisation
    -> suppression stopwords FR + EN
    -> ajout stopword métier : "générique"
    -> stemming SnowballStemmer
    -> TF-IDF
    -> Application du modèle
""",
    language="python"
)

st.markdown(
    """
La représentation finale repose sur **TF-IDF**, un choix cohérent pour les modèles classiques testés
dans le projet comme **KNN**, **Naive Bayes**, **SVM linéaire** et **XGBoost**. Le rapport final précise
également qu’un `LabelEncoder` a été utilisé pour transformer `prdtypecode` en 27 labels numériques.
"""
)

st.markdown("---")
st.header("3. Pré-processing des images")

st.subheader("Objectif du pipeline image")

st.markdown(
    """
Le pré-processing image vise à produire un flux d’images homogène, compatible avec le réseau
de neurones **EfficientNetB0**. Le rapport final indique que les images du dataset sont déjà très cohérentes :
elles sont en **JPEG**, en **RGB** et de taille **500x500 px**. Cela simplifie fortement la préparation
des données visuelles.
"""
)

st.subheader("Étapes appliquées sur les images")

image_steps = [
    "Lecture des métadonnées des images puis stockage dans un DataFrame",
    "Reconstruction du nom de fichier image à partir de `imageid` et `productid`",
    "Ajout des colonnes contenant le nom et le chemin des images",
    "Vérification de l’existence réelle des fichiers",
    "Redimensionnement des images en `224 x 224`",
    "Conversion des images de BGR vers RGB avec OpenCV",
    "Conversion des pixels en `float32`",
    "Application du prétraitement spécifique à EfficientNetB0 avec `preprocess_input()`",
    "Gestion des images manquantes ou illisibles par remplacement avec une image noire"
]

for step in image_steps:
    st.markdown(f"- {step}")

st.subheader("Pourquoi 224 x 224 ?")

st.markdown(
    """
Le redimensionnement en **224 x 224** correspond au format d’entrée attendu par **EfficientNetB0**.
Cette étape permet d’uniformiser toutes les images avant inférence, tout en restant compatible avec
les poids pré-entraînés sur **ImageNet** utilisés dans le projet.
"""
)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown(
        """
**Gestion des fichiers manquants**

Le rapport mentionne explicitement une vérification de l’existence des fichiers image.
Lorsqu’une image est absente ou illisible, elle est remplacée par une **image noire** afin de
préserver la structure des batchs et éviter les plantages dans le pipeline de Deep Learning.
"""
    )

with col_d:
    st.markdown(
        """
**Compatibilité avec EfficientNetB0**

La normalisation spécifique via `preprocess_input()` est indispensable pour rendre les données conformes
au pré-entraînement du modèle. Cela permet au réseau de recevoir des entrées dans le format attendu
et d’exploiter correctement les représentations visuelles apprises sur ImageNet.
"""
    )

st.markdown("---")
st.header("4. Résultat du pré-processing")

st.markdown(
    """
À l’issue de cette phase, deux pipelines propres et standardisés ont été obtenus :

- un **pipeline texte** prêt pour les modèles de classification classiques,
- un **pipeline image** prêt pour un modèle de Deep Learning basé sur EfficientNetB0.

Le pré-processing constitue donc une étape structurante du projet, car il conditionne directement
la qualité des représentations utilisées lors de la modélisation. Les choix effectués ont cherché
à concilier **simplicité**, **robustesse** et **compatibilité** avec les modèles retenus.
"""
)

st.markdown("---")
st.header("5. Synthèse")

col_x, col_y = st.columns(2)

with col_x:
    st.success(
        """
**Pipeline texte**
- Nettoyage HTML / URLs / ponctuation
- Minuscules
- Stopwords FR + EN
- Stopword métier : "générique"
- Stemming français
- TF-IDF
- Label encoding
"""
    )

with col_y:
    st.info(
        """
**Pipeline image**
- Reconstruction des chemins
- Vérification des fichiers
- Resize 224x224
- Conversion RGB
- Normalisation float32
- `preprocess_input()`
- Gestion des images manquantes
"""
    )
