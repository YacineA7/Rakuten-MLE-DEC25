from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Modèles", page_icon="🤖", layout="centered")

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.title("🤖 Modèles entraînés et résultats")
st.write(
    "La modélisation a été construite par étapes, avec des baselines texte, des modèles optimisés et un modèle deep learning sur les images. "
    "Cette page présente une synthèse des performances, une lecture détaillée par famille de modèles et une analyse des matrices de confusion.")


st.markdown("##  Stratégie progressive de modélisation")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        "### Baselines\n"
        "- **KNN** : utile comme point de comparaison, mais peu adapté à un espace TF-IDF très dimensionnel.\n" 
        "- **Naive Bayes** : rapide et simple, mais limité par son hypothèse d'indépendance."
    )
with col2:
    st.markdown(
        "### Modèles optimisés\n"
        "- **SVM linéaire** : très performant sur texte et intéressant pour l'interprétabilité. Inadapté aux très grands datasets.\n"
        "- **XGBoost** : meilleur score global grâce à une modélisation plus complexe des interactions."
    )
with col3:
    st.markdown(
        "### Deep Learning\n"
        "- **EfficientNetB0** : apporte une vraie valeur sur l'image, avec 77% d'accuracy.\n"
        "- Le résultat est d'autant plus intéressant que l'entraînement a été limité par les ressources machines."
    )

st.subheader("Synthèse des performances")
results_df = pd.DataFrame(
    {
        "Modèle": ["KNN", "Naive Bayes", "Naive Bayes optimisé", "SVM linéaire", "SVM optimisé", "XGBoost", "EfficientNetB0"],
        "Modalité": ["Texte", "Texte", "Texte", "Texte", "Texte", "Texte", "Image"],
        "Accuracy": ["73%", "69%", "76%", "74%", "76%", "80%", "77%"],
        "F1 pondéré": ["0.73", "0.67", "0.76", "0.74", "0.76", "0.809", "0.77"],
        "Lecture": [
            "Baseline basse performance",
            "Rapide mais naïf",
            "Gain modéré après optimisation",
            "Très bon classifieur texte",
            "Peu de gain malgré optimisation",
            "Meilleur modèle texte du projet",
            "Très bon modèle image malgré contraintes machine",
        ],
    }
)

st.dataframe(results_df, use_container_width=True, hide_index=True)

st.info(
    "La métrique principale retenue est le **F1-score pondéré**, car le problème est multiclasse et déséquilibré."
)


def show_confusion_matrix(filename: str, title: str, comment: str):
    path = FIGURES_DIR / filename
    st.markdown(f"### {title}")
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Matrice introuvable : `{filename}`")
    st.caption(comment)


st.markdown("## Matrices de confusion")
conf_tabs = st.tabs([
    "KNN",
    "Naive Bayes",
    "Naive Bayes optimisé",
    "SVM",
    "SVM optimisé",
    "XGBoost",
    "EfficientNetB0",
])

with conf_tabs[0]:
    show_confusion_matrix(
        "viz_txt_KNN_confusion.png",
        "KNN sur texte",
        "KNN sert de baseline basse performance. La matrice permet de visualiser ses confusions fréquentes dans un espace TF-IDF très dimensionnel."
    )

with conf_tabs[1]:
    show_confusion_matrix(
        "viz_txt_NB_confusion.png",
        "Naive Bayes multinomial",
        "Le modèle est rapide à entraîner, mais sa structure naïve limite sa capacité à séparer proprement certaines classes proches."
    )

with conf_tabs[2]:
    show_confusion_matrix(
        "viz_txt_NBOpt_confusion.png",
        "Naive Bayes multinomial optimisé",
        "L'optimisation améliore le score global, mais la matrice montre que certaines confusions persistent malgré le réglage des hyperparamètres."
    )

with conf_tabs[3]:
    show_confusion_matrix(
        "viz_txt_SVM_confusion.png",
        "SVM linéaire",
        "Le SVM est particulièrement pertinent pour la classification de texte. La matrice reflète une meilleure séparation de nombreuses classes que les baselines."
    )

with conf_tabs[4]:
    show_confusion_matrix(
        "viz_txt_SVMOpt_confusion.png",
        "SVM optimisé",
        "L'optimisation bayésienne apporte peu de gain sur les performances globales, ce qui confirme que le SVM de base était déjà très compétitif."
    )

with conf_tabs[5]:
    show_confusion_matrix(
        "viz_txt_XGB_confusion.png",
        "XGBoost",
        "XGBoost obtient les meilleurs résultats texte du projet. Cette matrice est centrale pour analyser les classes bien reconnues et les erreurs résiduelles."
    )

with conf_tabs[6]:

    show_confusion_matrix(
        "viz_img_ENB_confusion.png",
        "EfficientNetB0 - matrice de confusion",
        "La matrice montre la capacité du modèle image à distinguer les classes visuellement marquées, malgré un entraînement limité."
    )

    show_confusion_matrix(
        "viz_img_ENB_courbes.png",
        "EfficientNetB0 - courbes d'entraînement",
        "Les courbes permettent de suivre la convergence du modèle et d'évaluer l'absence de surapprentissage visible sur les epochs étudiées."
    )

st.markdown("## Lecture métier")
st.write(
    "Les matrices de confusion sont essentielles pour passer d'un score global à une compréhension métier des erreurs. "
    "Elles permettent d'identifier les catégories bien reconnues, les classes fréquemment confondues et les cas où un contrôle humain reste pertinent."
)
