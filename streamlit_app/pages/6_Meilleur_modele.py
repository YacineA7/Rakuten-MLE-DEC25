from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Meilleur modèle", page_icon="🏆", layout="centered")

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.title("🏆 Analyse du meilleur modèle")
st.write(
    "Le meilleur modèle du projet est **XGBoost sur les données textuelles**, avec **80% d'accuracy** et un **F1-score pondéré de 0.809**. "
    "Cette page en propose une lecture à la fois technique, interprétable et métier."
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Pourquoi XGBoost ressort")
    st.markdown(
        "- Meilleure performance globale parmi les modèles texte\n"
        "- Bonne capacité à modéliser des interactions complexes\n"
        "- Robuste dans un contexte multiclasse déséquilibré\n"
        "- Meilleur compromis performance globale / valeur opérationnelle"
    )

with col2:
    st.subheader("Point de vigilance")
    st.warning(
        "XGBoost est moins lisible qu'un SVM linéaire. Son interprétation nécessite donc des outils dédiés comme **LIME** pour expliquer localement les prédictions."
    )

st.markdown("## Matrice de confusion du meilleur modèle")
xgb_conf = FIGURES_DIR / "viz_txt_XGB_confusion.png"
if xgb_conf.exists():
    st.image(str(xgb_conf), use_container_width=True)
else:
    st.warning("Figure introuvable : `viz_txt_XGB_confusion.png`")

st.caption(
    "Cette matrice de confusion permet d'identifier les classes les mieux reconnues et les confusions résiduelles du meilleur modèle texte."
)

st.markdown("## Interprétabilité locale avec LIME")
lime_tabs = st.tabs(["KNN", "Naive Bayes", "SVM", "XGBoost"])
lime_files = {
    "KNN": "lime_knn_interpretation.png",
    "Naive Bayes": "lime_NB_interpretation.png",
    "SVM": "lime_SVM_interpretation.png",
    "XGBoost": "lime_XGB_interpretation.png",
}

lime_comments = {
    "KNN": "LIME montre quels mots poussent localement la prédiction KNN, malgré la faible interprétabilité globale du modèle.",
    "Naive Bayes": "L'interprétation aide à voir quels termes ont le plus contribué à la classe prédite dans une logique probabiliste simple.",
    "SVM": "Sur SVM, LIME permet d'illustrer les mots qui renforcent ou affaiblissent localement une décision linéaire.",
    "XGBoost": "Sur XGBoost, LIME est particulièrement utile pour expliquer localement un modèle performant mais moins transparent."
}

for tab, model_name in zip(lime_tabs, lime_files.keys()):
    with tab:
        path = FIGURES_DIR / lime_files[model_name]
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"Figure introuvable : `{lime_files[model_name]}`")
        st.caption(lime_comments[model_name])
        st.markdown(
            "- Les barres **vertes** renforcent la probabilité de la classe prédite\n"
            "- Les barres **rouges** diminuent cette probabilité"
        )

st.markdown("## Grad-CAM sur le modèle image")
gradcam_path = FIGURES_DIR / "gradcam_ENB_interpretation.png"
if gradcam_path.exists():
    st.image(str(gradcam_path), use_container_width=True)
else:
    st.warning("Figure introuvable : `gradcam_ENB_interpretation.png`")

st.caption(
    "Grad-CAM met en évidence les zones de l'image qui influencent le plus la prédiction d'EfficientNetB0."
)

st.markdown("## Lecture métier")
st.write(
    "L'interprétabilité est essentielle pour transformer un bon score en solution crédible. "
    "LIME aide à expliquer pourquoi un texte a conduit à une catégorie donnée, tandis que Grad-CAM montre ce que le modèle image regarde réellement. "
    "Ensemble, ces outils renforcent la confiance dans les prédictions et facilitent leur appropriation dans un cadre métier."
)

st.markdown("## Conclusion opérationnelle")
st.success(
    "Le meilleur modèle actuel est XGBoost sur texte, mais la présence d'un signal image interprétable ouvre naturellement vers une future architecture multimodale plus robuste."
)
