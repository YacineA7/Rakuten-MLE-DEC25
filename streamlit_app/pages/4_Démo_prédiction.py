import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Démo prédictions", page_icon="🔮", layout="wide")

BASE = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE / "models"
ARTIFACTS_DIR = BASE / "artifacts"

st.title("🔮 Démo prédictions")
st.markdown(
    """
Cette page exécute une **vraie prédiction** à partir des modèles enregistrés du projet.
Les erreurs observées venaient du fait que les modèles texte ont été entraînés sur du **TF-IDF vectorisé**,
et non directement sur une chaîne de caractères brute.
"""
)

st.info(
    "La page détecte maintenant automatiquement si l'objet chargé est un pipeline complet ou un modèle nécessitant une vectorisation TF-IDF préalable."
)

CLASS_LABELS = {
    10: "Livre / ouvrage imprimé",
    40: "Jeu vidéo / logiciel",
    50: "Accessoire / équipement technique",
    60: "Console / produit électronique",
    1140: "Figurine / produit dérivé",
    1160: "Carte / collection",
    1180: "Jeu / jouet enfant",
    1280: "Peluche / doudou / jouet textile",
    1281: "Linge / textile maison",
    1300: "Maquette / véhicule miniature",
    1301: "Papeterie / carnet / cahier",
    1302: "Sticker / décoration / support créatif",
    1320: "Produit bébé / enfant",
    1560: "Sac / emballage / contenant",
    1920: "Coussin / déco textile",
    2060: "Objet déco / mini-accessoire",
    2220: "Animal / univers enfant",
    2280: "Document / revue / bulletin",
    2403: "Autocollant / bande / patch",
    2462: "Accessoire mode / pratique",
    2522: "Fourniture bureautique",
    2582: "Mobilier / maison",
    2583: "Équipement maison / chauffage",
    2585: "Objet de collection / support imprimé",
    2705: "Livre jeunesse / BD / tome",
    2905: "Objet divers marketplace",
}

MODEL_FILES = {
    "KNN": "knn_text_model.joblib",
    "Naive Bayes": "nb_text_model.joblib",
    "SVM": "svm_text_model.joblib",
    "XGBoost": "xgb_text_model.joblib",
}

VECTORIZER_CANDIDATES = [
    MODELS_DIR / "tfidf_vectorizer.joblib",
    MODELS_DIR / "tfidf.joblib",
    MODELS_DIR / "vectorizer.joblib",
    ARTIFACTS_DIR / "tfidf_vectorizer.joblib",
    ARTIFACTS_DIR / "tfidf.joblib",
    ARTIFACTS_DIR / "vectorizer.joblib",
]

ENCODER_CANDIDATES = [
    MODELS_DIR / "label_encoder.joblib",
    MODELS_DIR / "le.joblib",
    ARTIFACTS_DIR / "label_encoder.joblib",
    ARTIFACTS_DIR / "le.joblib",
]


@st.cache_resource
def load_assets():
    loaded_models = {}
    model_errors = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        try:
            loaded_models[name] = joblib.load(path)
        except Exception as e:
            model_errors[name] = str(e)

    vectorizer, vectorizer_path, vectorizer_error = None, None, None
    for path in VECTORIZER_CANDIDATES:
        if path.exists():
            try:
                vectorizer = joblib.load(path)
                vectorizer_path = path
                break
            except Exception as e:
                vectorizer_error = str(e)

    label_encoder, label_encoder_path = None, None
    for path in ENCODER_CANDIDATES:
        if path.exists():
            try:
                label_encoder = joblib.load(path)
                label_encoder_path = path
                break
            except Exception:
                pass

    enb_exists = (MODELS_DIR / "ENB_model.keras").exists()
    return loaded_models, model_errors, vectorizer, vectorizer_path, vectorizer_error, label_encoder, label_encoder_path, enb_exists


models, model_errors, vectorizer, vectorizer_path, vectorizer_error, label_encoder, label_encoder_path, enb_exists = load_assets()


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("&amp;", " and ")
    text = text.replace("&nbsp;", " ")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text(title: str, description: str) -> str:
    return clean_text(f"{title} {description}")


def decode_label(raw_label):
    try:
        val = raw_label.item() if hasattr(raw_label, "item") else raw_label
    except Exception:
        val = raw_label

    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([int(val)])[0]
            code = int(decoded)
            return code, CLASS_LABELS.get(code, "Classe inconnue")
        except Exception:
            pass

    try:
        code = int(val)
        return code, CLASS_LABELS.get(code, "Classe inconnue")
    except Exception:
        return str(val), "Classe inconnue"


def needs_vectorization(model):
    name = model.__class__.__name__.lower()
    module = model.__class__.__module__.lower()
    if "pipeline" in name:
        return False
    if any(k in module for k in ["sklearn.neighbors", "sklearn.naive_bayes", "sklearn.svm", "xgboost"]):
        return True
    return True


def prepare_input(model, text):
    if not needs_vectorization(model):
        return [text], "pipeline intégré"
    if vectorizer is None:
        raise RuntimeError(
            "Aucun vectorizer TF-IDF sérialisé n'a été trouvé. Il faut ajouter par exemple tfidf_vectorizer.joblib dans ../models/ ou ../artifacts/."
        )
    X_vec = vectorizer.transform([text])
    return X_vec, f"vectorisation externe via {vectorizer_path.name}"


def top_predictions(model, X_input, top_k=5):
    pred = model.predict(X_input)[0]
    code, label = decode_label(pred)

    ranking = pd.DataFrame([{"prdtypecode": code, "classe": label, "probabilité": None}])

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_input)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                order = np.argsort(proba)[::-1][:top_k]
                rows = []
                for idx in order:
                    c, lab = decode_label(classes[idx])
                    rows.append(
                        {
                            "prdtypecode": c,
                            "classe": lab,
                            "probabilité": round(float(proba[idx]), 4),
                        }
                    )
                ranking = pd.DataFrame(rows)
        except Exception:
            pass

    return code, label, ranking


st.sidebar.header("État des artefacts")
for name in MODEL_FILES:
    if name in models:
        st.sidebar.success(f"{name} chargé")
    else:
        st.sidebar.error(f"{name} indisponible")

if vectorizer is not None:
    st.sidebar.success(f"TF-IDF chargé : {vectorizer_path.name}")
else:
    st.sidebar.warning("Aucun TF-IDF sérialisé détecté")
    if vectorizer_error:
        st.sidebar.caption(vectorizer_error)

if label_encoder is not None:
    st.sidebar.info(f"Label encoder chargé : {label_encoder_path.name}")
else:
    st.sidebar.caption("Aucun label encoder sérialisé détecté")

if enb_exists:
    st.sidebar.info("EfficientNetB0 détecté : ENB_model.keras")

if model_errors:
    with st.sidebar.expander("Détails des erreurs de chargement"):
        for name, err in model_errors.items():
            st.write(f"**{name}** : {err}")

available_models = list(models.keys())
if not available_models:
    st.error("Aucun modèle texte n'a pu être chargé depuis ../models/.")
    st.stop()

selected_model = st.selectbox("Choisir un modèle", available_models)

col_left, col_right = st.columns([1.15, 0.85])

with col_left:
    st.subheader("Entrées utilisateur")
    title = st.text_input("Titre / désignation du produit", value="Radiateur mural électrique 1500W")
    description = st.text_area(
        "Description du produit",
        value="Convecteur de chauffage pour maison, installation facile.",
        height=140,
    )

    preset = st.selectbox(
        "Exemple rapide",
        ["Aucun", "Carnet / papeterie", "Peluche enfant", "Radiateur chauffage", "Figurine collector", "Livre / tome", "Coussin déco"],
    )

    if preset != "Aucun":
        presets = {
            "Carnet / papeterie": ("Carnet A5 noir couverture rigide", "Bloc note de bureau avec pages lignées, format pratique pour prise de notes."),
            "Peluche enfant": ("Peluche hippopotame beige 25 cm", "Doudou doux pour enfant, jouet textile lavable."),
            "Radiateur chauffage": ("Radiateur mural électrique 1500W", "Convecteur de chauffage pour maison, installation facile."),
            "Figurine collector": ("Figurine collector manga édition limitée", "Statuette décorative issue d'un univers animé."),
            "Livre / tome": ("Tome 3 du roman jeunesse illustré", "Livre relié pour enfant, nouvelle édition."),
            "Coussin déco": ("Coussin décoratif imprimé lèvres rouges", "Housse textile maison pour canapé et décoration intérieure."),
        }
        title, description = presets[preset]
        st.text_input("Titre appliqué", value=title, disabled=True)
        st.text_area("Description appliquée", value=description, height=100, disabled=True)

with col_right:
    st.subheader("Image produit")
    uploaded_file = st.file_uploader("Dépose une image (optionnel)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_container_width=True)
    else:
        sample_img = BASE / "viz_image_grid-5.jpg"
        if sample_img.exists():
            st.image(sample_img.as_posix(), caption="Exemple visuel du corpus", use_container_width=True)

predict_btn = st.button("Lancer la prédiction", type="primary", use_container_width=True)

if predict_btn:
    text = build_text(title, description)
    model = models[selected_model]

    try:
        X_input, prep_mode = prepare_input(model, text)
        code, label, ranking = top_predictions(model, X_input)
    except Exception as e:
        st.error(f"Erreur pendant la prédiction avec {selected_model} : {e}")
        st.stop()

    st.markdown("---")
    st.subheader("Résultat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Classe prédite", str(code))
    c2.metric("Libellé métier", label)
    c3.metric("Modèle utilisé", selected_model)
    st.success(f"Prédiction du modèle : **{code} — {label}**")

    st.markdown("### Top classes proposées")
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    explain_col, pipe_col = st.columns(2)
    with explain_col:
        st.markdown("### Texte envoyé au pipeline")
        st.code(text if text else "(vide)", language="text")

    with pipe_col:
        st.markdown("### Prétraitement appliqué")
        st.write(f"Mode d'entrée : **{prep_mode}**")
        if needs_vectorization(model):
            st.write("Le texte brut est transformé en matrice TF-IDF avant appel au modèle.")
        else:
            st.write("Le modèle chargé intègre déjà son propre prétraitement.")

st.markdown("---")
with st.expander("Voir les classes du projet"):
    df_classes = pd.DataFrame([{"prdtypecode": k, "libellé": v} for k, v in CLASS_LABELS.items()]).sort_values("prdtypecode")
    st.dataframe(df_classes, use_container_width=True, hide_index=True)

st.caption("Page 4 - Démo prédictions | Version corrigée avec vectorisation TF-IDF si nécessaire")
