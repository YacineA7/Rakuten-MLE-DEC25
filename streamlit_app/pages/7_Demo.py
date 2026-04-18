from pathlib import Path
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st

from utils.model_loader import (
    TEXT_MODEL_FILES,
    IMAGE_MODEL_FILE,
    VECTORIZER_FILE,
    LABEL_ENCODER_FILE,
    load_joblib_artifact,
    load_keras_artifact,
    decode_prediction,
)
from utils.preprocessing import preprocess_text, preprocess_image

st.set_page_config(page_title="Démo", page_icon="🧪", layout="wide")

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "image_train"
DICTIONARY_FILE = DATA_DIR / "dictionnaire.csv"

TEXT_EXAMPLES = [
    {
        "nom": "Peluche Pokémon",
        "designation": "Peluche Pokémon Pikachu 30 cm",
        "description": "Peluche douce et lavable pour enfant, matière polyester, coloris jaune, idéale pour cadeau ou décoration de chambre.",
    },
    {
        "nom": "Jeu de cartes",
        "designation": "Booster Pokémon Écarlate et Violet",
        "description": "Booster neuf sous blister contenant 10 cartes officielles, version française, produit de collection.",
    },
    {
        "nom": "Carnet scolaire",
        "designation": "Carnet spirale A5 couverture rigide noir",
        "description": "Carnet ligné 200 pages, papier 90 g, adapté à la prise de notes, usage bureau ou scolaire.",
    },
    {
        "nom": "Taie d'oreiller",
        "designation": "Taie d'oreiller coton 65x65 cm blanche",
        "description": "Taie d'oreiller en coton doux, lavable en machine, format standard, pour literie adulte.",
    },
    {
        "nom": "Jeu vidéo",
        "designation": "Jeu vidéo FIFA 23 PS4",
        "description": "Jeu de football pour console PlayStation 4, version française, boîtier d'origine.",
    },
    {
        "nom": "Cahier enfants",
        "designation": "Cahier de coloriage animaux pour enfants",
        "description": "Livre de coloriage avec motifs animaux, grand format, papier épais, activité créative pour enfants.",
    },
]

IMAGE_EXAMPLES = [
    {"nom": "Exemple image 1", "filename": "image_99638230_product_1080207.jpg"},
    {"nom": "Exemple image 2", "filename": "image_1008141237_product_436067568.jpg"},
    {"nom": "Exemple image 3", "filename": "image_938777978_product_201115110.jpg"},
    {"nom": "Exemple image 4", "filename": "image_180193130_product_5612229.jpg"},
    {"nom": "Exemple image 5", "filename": "image_447423978_product_50310217.jpg"},
]


@st.cache_data
def load_category_mapping():
    if not DICTIONARY_FILE.exists():
        return {}

    df = pd.read_csv(DICTIONARY_FILE, encoding="utf-8")
    normalized_columns = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=normalized_columns)

    code_col = None
    label_col = None

    for col in df.columns:
        if "prdtypecode" in col or col == "code":
            code_col = col
        if any(token in col for token in ["categorie", "catégorie", "libelle", "label"]):
            label_col = col

    if code_col is None or label_col is None:
        return {}

    return dict(zip(df[code_col].astype(str), df[label_col].astype(str)))


@st.cache_resource
def load_shared_artifacts():
    vectorizer = None
    label_encoder = None

    vectorizer_path = MODELS_DIR / VECTORIZER_FILE
    label_encoder_path = MODELS_DIR / LABEL_ENCODER_FILE

    if vectorizer_path.exists():
        vectorizer = load_joblib_artifact(vectorizer_path)
    if label_encoder_path.exists():
        label_encoder = load_joblib_artifact(label_encoder_path)

    return vectorizer, label_encoder


@st.cache_resource
def load_text_model(model_name: str):
    model_path = MODELS_DIR / TEXT_MODEL_FILES[model_name]
    return load_joblib_artifact(model_path)


@st.cache_resource
def load_image_model():
    model_path = MODELS_DIR / IMAGE_MODEL_FILE
    return load_keras_artifact(model_path)


def format_category(code, category_mapping):
    code_str = str(code)
    label = category_mapping.get(code_str)
    if label:
        return f"{code_str} — {label}"
    return code_str


def safe_predict_text(model, vectorizer, label_encoder, text: str, category_mapping: dict):
    X = vectorizer.transform([text])
    expected = getattr(model, "n_features_in_", None)
    got = X.shape[1]

    if expected is not None and expected != got:
        raise ValueError(
            f"Incompatibilité modèle/vectorizer : le modèle attend {expected} features, "
            f"mais le vectorizer en produit {got}."
        )

    pred = model.predict(X)[0]
    label = decode_prediction(pred, label_encoder)

    top_classes = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        top_idx = np.argsort(proba)[::-1][:3]
        top_classes = [
            {
                "classe": str(decode_prediction(int(i), label_encoder)),
                "classe_affichee": format_category(decode_prediction(int(i), label_encoder), category_mapping),
                "proba": float(proba[i]),
            }
            for i in top_idx
        ]

    return label, format_category(label, category_mapping), top_classes


def safe_predict_image(model, label_encoder, img_array, category_mapping: dict):
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    label = decode_prediction(pred_idx, label_encoder)
    top_idx = np.argsort(preds)[::-1][:3]
    top_classes = [
        {
            "classe": str(decode_prediction(int(i), label_encoder)),
            "classe_affichee": format_category(decode_prediction(int(i), label_encoder), category_mapping),
            "proba": float(preds[i]),
        }
        for i in top_idx
    ]
    return label, format_category(label, category_mapping), top_classes


def image_example_to_upload(image_path: Path):
    with open(image_path, "rb") as f:
        data = f.read()
    bio = BytesIO(data)
    bio.name = image_path.name
    return bio


def render_top_classes(top_classes):
    if not top_classes:
        st.info("Ce modèle ne fournit pas de probabilités.")
        return

    st.markdown("### Top 3 des catégories")
    for idx, item in enumerate(top_classes, start=1):
        st.write(f"{idx}. **{item['classe_affichee']}** — probabilité : `{item['proba']:.4f}`")


category_mapping = load_category_mapping()
vectorizer, label_encoder = load_shared_artifacts()

st.title("🧪 Démonstration des modèles")
st.markdown(
    "Cette page propose une démonstration guidée avec des **exemples prédéfinis** afin de fiabiliser la soutenance et de comparer rapidement les comportements des modèles."
)

mode = st.radio(
    "Choisir une modalité",
    ["Texte", "Image", "Comparaison"],
    horizontal=True,
)

if mode == "Texte":
    st.subheader("Prédiction texte")

    col1, col2 = st.columns([1, 1])
    with col1:
        example_name = st.selectbox(
            "Choisir un exemple texte",
            [ex["nom"] for ex in TEXT_EXAMPLES],
        )
        selected_model = st.selectbox("Choisir un modèle texte", list(TEXT_MODEL_FILES.keys()))
        use_stemming = st.toggle("Activer le stemming français", value=False)

    selected_example = next(ex for ex in TEXT_EXAMPLES if ex["nom"] == example_name)
    designation = selected_example["designation"]
    description = selected_example["description"]

    with col2:
        st.markdown("### Exemple sélectionné")
        st.write(f"**Désignation :** {designation}")
        st.write(f"**Description :** {description}")

    if st.button("Lancer la prédiction texte", use_container_width=True):
        model_path = MODELS_DIR / TEXT_MODEL_FILES[selected_model]

        if not model_path.exists():
            st.error(f"Modèle introuvable : {model_path.name}")
        elif vectorizer is None or label_encoder is None:
            st.error("Le vectorizer TF-IDF ou le label encoder n'ont pas été chargés.")
        else:
            try:
                model = load_text_model(selected_model)
                processed_text = preprocess_text(designation, description, use_stemming=use_stemming)
                pred_code, pred_display, top_classes = safe_predict_text(
                    model, vectorizer, label_encoder, processed_text, category_mapping
                )

                st.success(f"Catégorie prédite : **{pred_display}**")
                st.caption(f"Code prédit : {pred_code}")

                st.markdown("### Texte après préprocessing")
                st.code(processed_text, language="text")
                render_top_classes(top_classes)
            except Exception as e:
                st.exception(e)

elif mode == "Image":
    st.subheader("Prédiction image")

    available_image_examples = []
    for ex in IMAGE_EXAMPLES:
        img_path = IMAGES_DIR / ex["filename"]
        if img_path.exists():
            available_image_examples.append({**ex, "path": img_path})

    if not available_image_examples:
        st.error("Aucune image d'exemple n'a été trouvée dans le dossier data/images/image_train.")
    else:
        selected_name = st.selectbox(
            "Choisir une image exemple",
            [ex["nom"] for ex in available_image_examples],
        )
        selected_example = next(ex for ex in available_image_examples if ex["nom"] == selected_name)
        image_path = selected_example["path"]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(str(image_path), caption=image_path.name, width=320)
        with col2:
            st.write(f"**Exemple :** {selected_example['nom']}")
            st.write(f"**Fichier :** `{image_path.name}`")

        if st.button("Lancer la prédiction image", use_container_width=True):
            model_path = MODELS_DIR / IMAGE_MODEL_FILE
            if not model_path.exists():
                st.error(f"Modèle image introuvable : {model_path.name}")
            elif label_encoder is None:
                st.error("Le label encoder n'a pas été chargé.")
            else:
                try:
                    model = load_image_model()
                    upload_like = image_example_to_upload(image_path)
                    pil_image, img_array = preprocess_image(upload_like)
                    pred_code, pred_display, top_classes = safe_predict_image(
                        model, label_encoder, img_array, category_mapping
                    )

                    st.success(f"Catégorie prédite : **{pred_display}**")
                    st.caption(f"Code prédit : {pred_code}")
                    st.image(pil_image, caption="Image prétraitée (224x224 RGB)", width=224)
                    render_top_classes(top_classes)
                except Exception as e:
                    st.exception(e)

else:
    st.subheader("Comparaison des modèles texte")

    col1, col2 = st.columns([1, 1])
    with col1:
        example_name = st.selectbox(
            "Choisir un exemple pour comparer les modèles",
            [ex["nom"] for ex in TEXT_EXAMPLES],
            key="compare_example",
        )
        use_stemming = st.toggle("Activer le stemming pour la comparaison", value=False)

    selected_example = next(ex for ex in TEXT_EXAMPLES if ex["nom"] == example_name)
    designation = selected_example["designation"]
    description = selected_example["description"]

    with col2:
        st.markdown("### Exemple sélectionné")
        st.write(f"**Désignation :** {designation}")
        st.write(f"**Description :** {description}")

    if st.button("Comparer les modèles", use_container_width=True):
        if vectorizer is None or label_encoder is None:
            st.error("Le vectorizer TF-IDF ou le label encoder n'ont pas été chargés.")
        else:
            processed_text = preprocess_text(designation, description, use_stemming=use_stemming)
            rows = []

            for model_name, file_name in TEXT_MODEL_FILES.items():
                model_path = MODELS_DIR / file_name
                if not model_path.exists():
                    rows.append(
                        {
                            "Modèle": model_name,
                            "Code prédit": "Fichier absent",
                            "Catégorie": "-",
                            "Top 1 probabilité": "-",
                        }
                    )
                    continue

                try:
                    model = load_text_model(model_name)
                    pred_code, pred_display, top_classes = safe_predict_text(
                        model, vectorizer, label_encoder, processed_text, category_mapping
                    )
                    top1 = f"{top_classes[0]['proba']:.4f}" if top_classes else "N/A"
                    rows.append(
                        {
                            "Modèle": model_name,
                            "Code prédit": str(pred_code),
                            "Catégorie": pred_display,
                            "Top 1 probabilité": top1,
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "Modèle": model_name,
                            "Code prédit": "Erreur",
                            "Catégorie": str(e),
                            "Top 1 probabilité": "-",
                        }
                    )

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("### Texte après préprocessing")
            st.code(processed_text, language="text")
