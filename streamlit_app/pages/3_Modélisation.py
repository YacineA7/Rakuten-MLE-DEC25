import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Modélisation", page_icon="📈", layout="wide")

BASE = Path(__file__).resolve().parents[2]

st.title("📈 Modélisation")
st.markdown(
    """
Cette page présente les principaux modèles testés dans le projet Rakuten, leurs performances,
ainsi que les enseignements méthodologiques tirés des phases baseline, optimisation et deep learning.
"""
)

# ---------- Données synthétiques issues du rapport / notebooks ----------
results = pd.DataFrame([
    {
        "Modèle": "KNN (texte TF-IDF)",
        "Famille": "Baseline texte",
        "Accuracy": 0.76,
        "F1-macro": 0.75,
        "F1-pondéré": 0.77,
        "Forces": "Simple, robuste, bon point de départ",
        "Limites": "Lent à grande échelle, peu interprétable nativement"
    },
    {
        "Modèle": "Naive Bayes multinomial",
        "Famille": "Baseline texte",
        "Accuracy": 0.74,
        "F1-macro": 0.73,
        "F1-pondéré": 0.74,
        "Forces": "Très rapide, efficace sur texte sparse",
        "Limites": "Hypothèse d'indépendance forte"
    },
    {
        "Modèle": "SVM linéaire",
        "Famille": "Texte optimisé",
        "Accuracy": 0.79,
        "F1-macro": 0.78,
        "F1-pondéré": 0.79,
        "Forces": "Très bon compromis précision / stabilité",
        "Limites": "Coût d'entraînement plus élevé"
    },
    {
        "Modèle": "XGBoost",
        "Famille": "Boosting texte",
        "Accuracy": 0.80,
        "F1-macro": 0.798,
        "F1-pondéré": 0.809,
        "Forces": "Capture des interactions plus complexes",
        "Limites": "Moins interprétable, tuning nécessaire"
    },
    {
        "Modèle": "EfficientNetB0 (images)",
        "Famille": "Deep Learning image",
        "Accuracy": 0.77,
        "F1-macro": 0.76,
        "F1-pondéré": 0.77,
        "Forces": "Très bon niveau sur image seule",
        "Limites": "Coût GPU / temps d'entraînement"
    },
])

# ---------- KPI ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nombre de modèles", len(results))
col2.metric("Meilleure accuracy", f"{results['Accuracy'].max():.1%}")
col3.metric("Meilleur F1 pondéré", f"{results['F1-pondéré'].max():.1%}")
col4.metric("Meilleur modèle", results.sort_values('F1-pondéré', ascending=False).iloc[0]['Modèle'])

st.markdown("---")

# ---------- Tableau résultats ----------
st.subheader("Tableau de synthèse")
st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)

# ---------- Comparaison graphique ----------
st.subheader("Comparaison des performances")
chart_df = results[["Modèle", "Accuracy", "F1-macro", "F1-pondéré"]].set_index("Modèle")
st.bar_chart(chart_df)

# ---------- Analyse méthodologique ----------
st.subheader("Lecture des résultats")
left, right = st.columns([1.1, 0.9])

with left:
    st.markdown(
        """
### Ce qu'il faut retenir
- **Les modèles texte dominent globalement** grâce à la richesse sémantique des titres et descriptions.
- **XGBoost** obtient ici la meilleure performance globale sur la partie texte.
- **SVM linéaire** offre un excellent compromis entre performance, robustesse et simplicité de mise en production.
- **EfficientNetB0** montre que l'image seule est déjà très informative, malgré un entraînement sur échantillon réduit.
- **Naive Bayes** reste une baseline rapide et utile pour cadrer le niveau minimal attendu.
        """
    )

with right:
    st.info(
        """
**Conclusion projet**

Le texte reste la modalité la plus discriminante.
L'image apporte une valeur complémentaire importante,
et ouvre naturellement vers une future approche multimodale.
        """
    )

st.markdown("---")

# ---------- Focus modèles ----------
st.subheader("Focus par modèle")

tabs = st.tabs(["KNN", "Naive Bayes", "SVM", "XGBoost", "EfficientNetB0"])

with tabs[0]:
    st.markdown(
        """
### KNN
Le KNN a servi de **baseline texte** simple à implémenter. Il fournit un niveau de performance correct,
mais devient coûteux en prédiction sur de gros volumes.

**Pourquoi il est utile :**
- facile à expliquer conceptuellement,
- bon repère initial,
- compatible avec une analyse locale de type SHAP/LIME.
        """
    )

with tabs[1]:
    st.markdown(
        """
### Naive Bayes multinomial
Naive Bayes est particulièrement adapté aux représentations **TF-IDF sparse**. Il est très rapide
et constitue un excellent benchmark de départ.

**Points forts :**
- apprentissage rapide,
- très bon comportement sur texte,
- faible coût de calcul.
        """
    )

with tabs[2]:
    st.markdown(
        """
### SVM linéaire
Le SVM linéaire est l'un des meilleurs compromis du projet pour la classification textuelle multi-classes.
Il gère bien les espaces de grande dimension issus de TF-IDF.

**Pourquoi il performe bien :**
- séparation efficace dans un espace sparse,
- bonne robustesse au déséquilibre avec `class_weight='balanced'`,
- résultats stables.
        """
    )

with tabs[3]:
    st.markdown(
        """
### XGBoost
XGBoost apporte une modélisation plus riche grâce au boosting successif d'arbres de décision.
Il permet de capturer des interactions plus fines entre variables textuelles.

**En pratique :**
- meilleure métrique globale observée,
- bonne capacité de généralisation,
- coût de tuning plus important.
        """
    )

with tabs[4]:
    st.markdown(
        """
### EfficientNetB0
EfficientNetB0 a été utilisé sur la modalité image pour évaluer le potentiel du **deep learning visuel**.
Le modèle montre que l'image seule permet déjà de bonnes performances.
        """
    )

    img_path = BASE / "viz_ENB_courbe-2.jpg"
    if img_path.exists():
        st.image(Image.open(img_path), caption="Courbes d'apprentissage EfficientNetB0", use_container_width=True)
        st.success(
            "Les courbes montrent une convergence régulière, sans surapprentissage marqué sur la période observée."
        )
    else:
        st.warning("Visualisation EfficientNetB0 introuvable dans le projet.")

st.markdown("---")

# ---------- Pipeline ----------
st.subheader("Pipeline de modélisation")
st.markdown(
    """
1. **Préparation du texte** : nettoyage HTML, minuscules, suppression ponctuation/chiffres.
2. **Pré-traitement linguistique** : stopwords FR/EN, stemming.
3. **Vectorisation** : TF-IDF avec unigrammes et bigrammes.
4. **Split stratifié** : conservation de la distribution des 27 classes.
5. **Entraînement** des modèles baseline puis optimisés.
6. **Évaluation** via Accuracy, F1-macro, F1-pondéré et matrices de confusion.
7. **Interprétabilité** : LIME / SHAP pour le texte, Grad-CAM pour l'image.
    """
)

# ---------- Recommandation ----------
st.subheader("Recommandation de mise en production")
choice = st.selectbox(
    "Choisir un scénario cible",
    [
        "Production simple et robuste",
        "Meilleure performance texte",
        "Vision multimodale future"
    ]
)

if choice == "Production simple et robuste":
    st.success("Recommandation : **SVM linéaire** pour son excellent compromis entre performance, stabilité et simplicité.")
elif choice == "Meilleure performance texte":
    st.success("Recommandation : **XGBoost** si l'objectif prioritaire est le score maximal sur la modalité texte.")
else:
    st.success("Recommandation : combiner **modèle texte + modèle image** dans une future architecture multimodale.")

st.caption("Page 3 - Modélisation | Projet Rakuten France Multimodal Product Data Classification")
