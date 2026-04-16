import streamlit as st
import joblib
import tensorflow as tf

st.set_page_config(
    page_title="Rakuten Product Classifier",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Classification de produits Rakuten")
st.write("Projet de Data Science — Classification multiclasse texte + image")

st.info("Navigue dans les onglets à gauche pour explorer le projet.")

@st.cache_resource
def load_models():
    knn_model = joblib.load("../models/knn_text_model.joblib")
    svm_model = joblib.load("../models/svm_text_model.joblib")
    xgb_model = joblib.load("../models/xgb_text_model.joblib")
    nb_model = joblib.load("../models/nb_text_model.joblib")

    ENB = tf.keras.models.load_model("../models/ENB_model.keras")
    # return knn_model, svm_model, xgb_model, ENB
    return knn_model

knn_model = load_models()

st.title("Contexte et périmètre du projet")
st.markdown(
    """
Le projet est proposé par l’ENS Paris, dans le cadre du challenge Rakuten France Multimodal Product Data Classification (Challenge  Data). 
L’objectif est de créer un système permettant d’automatiser, en exploitant des données textuelles et des données images,
le catalogage des produits e-commerce de la plateforme, donc de prédire le code de type produit, à partir de ces données.

"""
)

st.title("Enjeux métier")
st.markdown(
    """
La classification automatique des produits est un sujet central pour toutes les marketplaces e-commerce. Elle permet de:
1 - Améliorer la recherche: en labellisant les produits dans la bonne catégorie, les résultats et recommandations deviennent plus pertinents.
2 - Automatiser le catalogage: éviter les erreurs humaines lors de la saisie des articles
3 - Détecter les anomalies: limiter la frustration des clients dû à des produits classés dans la mauvaise catégorie, et qui ne correspondent pas à leur recherche
4 - Optimiser la navigation: les filtres de catégories sont efficaces seulement si les produits sont correctement classifiés.

"""
)