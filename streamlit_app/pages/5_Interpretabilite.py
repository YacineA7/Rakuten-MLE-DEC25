from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Interprétabilité", page_icon="🧠", layout="wide")

BASE = Path(__file__).resolve().parents[2]

st.title("🧠 Interprétabilité")
st.markdown(
    """
Cette page présente la logique d'interprétation des prédictions dans le projet Rakuten.
L'objectif est de rendre les modèles plus lisibles en expliquant **pourquoi** une classe produit est prédite,
à partir du texte ou de l'image.
"""
)

st.info(
    "Dans le rapport final, l'interprétabilité texte a été réalisée avec **LIME**, tandis que l'interprétabilité image a été réalisée avec **Grad-CAM** sur EfficientNetB0."
)

# ------------------------------------------------------------------
# Résumé méthodologique
# ------------------------------------------------------------------
st.subheader("Méthodes utilisées")
methods = pd.DataFrame([
    {
        "Modalité": "Texte",
        "Méthode": "LIME",
        "Modèles concernés": "KNN, Naive Bayes, SVM, XGBoost",
        "But": "Identifier les mots qui augmentent ou diminuent la probabilité de la classe prédite"
    },
    {
        "Modalité": "Texte",
        "Méthode": "SHAP (extension possible)",
        "Modèles concernés": "KNN, XGBoost, SVM",
        "But": "Quantifier la contribution locale des variables à la prédiction"
    },
    {
        "Modalité": "Image",
        "Méthode": "Grad-CAM",
        "Modèles concernés": "EfficientNetB0",
        "But": "Mettre en évidence les zones visuelles les plus influentes dans la décision du CNN"
    },
])
st.dataframe(methods, use_container_width=True, hide_index=True)

st.markdown("---")

# ------------------------------------------------------------------
# Interprétation texte
# ------------------------------------------------------------------
st.subheader("Interprétabilité texte")
left, right = st.columns([1.1, 0.9])

with left:
    st.markdown(
        """
### Principe
Pour les modèles texte, le projet s'appuie d'abord sur **LIME** pour expliquer localement une prédiction.
L'idée est simple : perturber le texte d'entrée, observer la variation de la prédiction, puis estimer
quels mots poussent le modèle vers une classe donnée.

### Lecture métier
- Les contributions **positives** renforcent la classe prédite.
- Les contributions **négatives** vont contre cette classe.
- Les mots les plus spécifiques au produit sont généralement les plus discriminants.
        """
    )

with right:
    st.success(
        "Exemple d'interprétation : un texte contenant des termes comme `carnet`, `bloc note`, `spirale` ou `agenda` orientera fortement la décision vers une classe de papeterie."
    )
    st.warning(
        "Limite importante : une explication locale ne résume pas tout le comportement global du modèle."
    )

model_choice = st.selectbox(
    "Choisir un modèle texte à expliquer",
    ["KNN", "Naive Bayes", "SVM", "XGBoost"]
)

example_map = {
    "KNN": {
        "classe": "1301",
        "resume": "Le KNN s'appuie sur la proximité entre textes vectorisés. Les mots saillants rapprochent l'exemple de voisins de même catégorie."
    },
    "Naive Bayes": {
        "classe": "2060",
        "resume": "Naive Bayes donne un poids fort aux termes les plus probables dans une classe donnée, ce qui rend l'effet de certains mots très visible."
    },
    "SVM": {
        "classe": "1301",
        "resume": "Le SVM linéaire apprend une frontière dans l'espace TF-IDF ; les mots ayant les coefficients les plus élevés pèsent fortement dans la décision."
    },
    "XGBoost": {
        "classe": "1301",
        "resume": "XGBoost capture des interactions plus complexes entre mots, mais cela le rend moins directement interprétable qu'un modèle linéaire."
    },
}

st.markdown(f"### Focus {model_choice}")
st.write(example_map[model_choice]["resume"])
st.metric("Exemple de classe expliquée dans le rapport", example_map[model_choice]["classe"])

user_text = st.text_area(
    "Tester une explication textuelle simplifiée",
    value="Carnet A5 noir spirales 120 pages bloc note bureau",
    height=120,
)

keywords = {
    "1301": ["carnet", "bloc", "note", "agenda", "spirales", "pages"],
    "1280": ["peluche", "doudou", "nounours"],
    "2583": ["radiateur", "chauffage", "convecteur"],
    "2705": ["tome", "livre", "roman", "manga"],
}

if st.button("Analyser le texte", use_container_width=True):
    text_lower = user_text.lower()
    hits = []
    for code, words in keywords.items():
        for w in words:
            if w in text_lower:
                hits.append((code, w))

    if hits:
        df_hits = pd.DataFrame(hits, columns=["Classe probable", "Mot déclencheur"])
        st.dataframe(df_hits, use_container_width=True, hide_index=True)
        st.success("Cette vue simplifiée illustre comment certains mots orientent fortement la prédiction locale.")
    else:
        st.info("Aucun mot fortement discriminant n'a été détecté dans cet exemple simplifié.")

st.markdown("---")

# ------------------------------------------------------------------
# SHAP section
# ------------------------------------------------------------------
st.subheader("Extension possible avec SHAP")
st.markdown(
    """
SHAP n'était pas la méthode principale du rapport final, mais elle constitue une **extension pertinente**
pour enrichir la démonstration d'interprétabilité, notamment sur **KNN** ou **XGBoost**.

### Pourquoi utiliser SHAP ?
- meilleure quantification des contributions locales,
- lecture plus standardisée des effets positifs et négatifs,
- utile pour une démonstration plus poussée en soutenance.
    """
)

code_example = '''
explainer = shap.KernelExplainer(knn.predict_proba, X_bg_dense)
shap_values = explainer.shap_values(X_explain_dense, nsamples=100)
shap.summary_plot(shap_values[class_idx], X_explain_dense, feature_names=feature_names)
'''
st.code(code_example, language="python")

st.caption("Pour KNN, SHAP est plus coûteux en calcul et doit être appliqué sur un petit sous-échantillon.")

st.markdown("---")

# ------------------------------------------------------------------
# Interprétation image
# ------------------------------------------------------------------
st.subheader("Interprétabilité image avec Grad-CAM")
col1, col2 = st.columns([0.95, 1.05])

with col1:
    st.markdown(
        """
### Principe
Grad-CAM permet de visualiser les régions de l'image qui influencent le plus la prédiction d'un CNN.
Dans le projet, cette approche a été utilisée sur **EfficientNetB0** afin d'identifier les zones
visuellement déterminantes pour la classe prédite.

### Intérêt métier
- vérifier que le modèle regarde bien le produit,
- détecter si la décision repose sur un fond parasite,
- renforcer la confiance dans la prédiction.
        """
    )

with col2:
    img_path = BASE / "viz_ENB_courbe-2.jpg"
    if img_path.exists():
        st.image(Image.open(img_path), caption="Courbes d'apprentissage EfficientNetB0", use_container_width=True)
    else:
        st.caption("Aucune visualisation locale disponible pour le moment.")

st.info(
    "Dans le rapport, Grad-CAM est présenté comme la méthode d'interprétabilité du modèle image, pour mettre en évidence les zones les plus influentes dans la décision finale."
)

# ------------------------------------------------------------------
# Comparaison
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("Comparer les approches")
comparison = pd.DataFrame([
    {
        "Méthode": "LIME",
        "Type": "Local",
        "Avantages": "Simple à expliquer, visuel, adapté à la soutenance",
        "Limites": "Approximation locale, stabilité variable"
    },
    {
        "Méthode": "SHAP",
        "Type": "Local / semi-global",
        "Avantages": "Contributions plus rigoureuses, lecture standardisée",
        "Limites": "Coût de calcul élevé sur modèles complexes ou KNN"
    },
    {
        "Méthode": "Grad-CAM",
        "Type": "Visuel local",
        "Avantages": "Très intuitif sur image, utile pour montrer où regarde le modèle",
        "Limites": "Dépend de l'architecture CNN, n'explique pas tout le pipeline"
    },
])
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.markdown(
    """
### Recommandation pour la soutenance
- **LIME** pour montrer rapidement l'interprétation des modèles texte,
- **SHAP** comme approfondissement méthodologique sur un modèle choisi,
- **Grad-CAM** pour la partie image, très convaincante visuellement.
    """
)

st.caption("Page 5 - Interprétabilité | Projet Rakuten France Multimodal Product Data Classification")
