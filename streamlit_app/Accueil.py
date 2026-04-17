from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Rakuten - Classification multimodale",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.subtitle {
    font-size: 1.15rem;
    color: #4b5563;
    margin-bottom: 1.2rem;
}
.hero-box {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    padding: 1.5rem 1.5rem;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.25rem 1.25rem;
    height: 100%;
}
.small-muted {
    color: #6b7280;
    font-size: 0.95rem;
}
.kpi {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.badge {
    display: inline-block;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    background: #111827;
    color: white;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.highlight {
    color: #1d4ed8;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.45, 1], gap="large")

with col1:
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    st.markdown('<div class="badge">Projet Data Scientist · Rakuten France</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-title">Classification multimodale de produits e-commerce</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Cette application présente notre démarche de bout en bout pour prédire la catégorie '
        'd’un produit Rakuten à partir de ses données textuelles et visuelles.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
Le projet répond à un enjeu central de marketplace : **automatiser le catalogage produit** afin
d’améliorer la qualité de la recherche, des recommandations, de la navigation par catégories et de
l’expérience utilisateur. Dans notre cas, la prédiction repose sur deux modalités complémentaires :
le **texte** (désignation, description) et l’**image produit**.
"""
    )

    st.markdown(
        """
Nous avons structuré l’étude en plusieurs étapes : **exploration des données**, **préprocessing**,
**modélisation classique sur le texte**, **deep learning sur les images**, puis **analyse métier**
des résultats. L’objectif de cette application est de rendre ces travaux lisibles, interactifs et
exploitables dans une logique de démonstration et de PoC.
"""
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎯 Problématique")
    st.write(
        "Comment classer automatiquement un produit Rakuten dans la bonne catégorie "
        "à partir de sa désignation, de sa description et de son image ?"
    )

    st.subheader("💼 Enjeux métier")
    st.markdown(
        "- Réduire les erreurs de catégorisation\n"
        "- Améliorer la recherche et les filtres\n"
        "- Fiabiliser les recommandations\n"
        "- Accélérer la mise en ligne des produits"
    )

    st.subheader("🧪 Modalités étudiées")
    st.markdown(
        "- **Texte** : désignation, description\n"
        "- **Image** : photo produit\n"
        "- **Vision projet** : vers une approche multimodale"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## Chiffres clés")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi">99k</div>', unsafe_allow_html=True)
    st.markdown("produits environ")
    st.markdown('<div class="small-muted">Jeu global train + test</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi">27</div>', unsafe_allow_html=True)
    st.markdown("catégories produits")
    st.markdown('<div class="small-muted">Variable cible : prdtypecode</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi">80.9%</div>', unsafe_allow_html=True)
    st.markdown("F1-score pondéré")
    st.markdown('<div class="small-muted">Meilleur résultat obtenu avec XGBoost texte</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with k4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi">77%</div>', unsafe_allow_html=True)
    st.markdown("accuracy image")
    st.markdown('<div class="small-muted">EfficientNetB0 sur la modalité visuelle</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## Vue d’ensemble")

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🔎 Ce que montre l’étude")
    st.markdown(
        "- Les données texte sont la source d’information la plus discriminante\n"
        "- Le jeu présente un **déséquilibre de classes** important\n"
        "- Les descriptions sont parfois manquantes ou peu informatives\n"
        "- Les images apportent une **information complémentaire utile**\n"
        "- Le meilleur compromis performance obtenu est **XGBoost sur le texte**"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🚀 Ce que permet l’application")
    st.markdown(
        "- Explorer les données et les visualisations clés\n"
        "- Comprendre les choix de preprocessing\n"
        "- Comparer les modèles entraînés et leurs performances\n"
        "- Tester une **démo de prédiction**\n"
        "- Illustrer un **PoC métier** de pré-catégorisation produit"
    )
    st.markdown('</div>', unsafe_allow_html=True)


st.caption(
    "Application de soutenance Streamlit du projet Rakuten - classification multimodale de produits e-commerce."
)