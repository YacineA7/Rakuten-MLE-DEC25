import streamlit as st

st.set_page_config(page_title="Sujet & enjeux", page_icon="🎯", layout="centered")

st.title("🎯 Sujet, problème et enjeux")
st.markdown(
    """
Cette page introduit le projet Rakuten de **classification multimodale de produits e-commerce**.  
L'objectif est de prédire automatiquement la catégorie d'un produit à partir de sa **désignation**, de sa **description** et de son **image**.
"""
)

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("Contexte du challenge")
    st.write(
        "Le projet s'inscrit dans le challenge Rakuten France Multimodal Product Data Classification. "
        "Il vise à automatiser le catalogage produit sur une marketplace e-commerce, en exploitant des données texte et image."
    )

    st.subheader("Problème de data science")
    st.info(
        "Prédire la variable cible `prdtypecode` parmi 27 catégories à partir de données hétérogènes, "
        "avec un jeu déséquilibré et une qualité textuelle variable."
    )

    st.subheader("Pourquoi ce sujet est important")
    st.markdown(
        "- La bonne catégorie améliore la recherche et les filtres\n"
        "- Elle renforce la pertinence des recommandations\n"
        "- Elle limite les erreurs humaines de catalogage\n"
        "- Elle réduit la frustration liée aux produits mal classés"
    )

with col2:
    st.subheader("Vision métier")
    st.success(
        "Un modèle de classification fiable peut servir d'assistant de pré-catégorisation pour aider les vendeurs "
        "ou automatiser une partie du contrôle qualité catalogue."
    )

    st.subheader("Question centrale")
    st.markdown(
        "**Comment exploiter efficacement texte et image pour classer automatiquement un produit Rakuten,** "
        "tout en gardant une approche compréhensible, mesurable et utile côté métier ?"
    )

    st.subheader("Objectifs de l'application")
    st.markdown(
        "- Présenter la démarche du projet\n"
        "- Montrer les analyses exploratoires\n"
        "- Comparer les modèles\n"
        "- Expliquer le meilleur modèle\n"
        "- Démontrer une application métier concrète"
    )
