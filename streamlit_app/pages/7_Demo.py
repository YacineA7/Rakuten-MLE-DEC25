from pathlib import Path
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


def safe_predict_text(model, vectorizer, label_encoder, text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = decode_prediction(pred, label_encoder)

    top_classes = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        top_idx = np.argsort(proba)[::-1][:3]
        top_classes = [
            {
                "classe": str(decode_prediction(int(i), label_encoder)),
                "proba": float(proba[i])
            }
            for i in top_idx
        ]
    return label, top_classes


def safe_predict_image(model, label_encoder, img_array):
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    label = decode_prediction(pred_idx, label_encoder)
    top_idx = np.argsort(preds)[::-1][:3]
    top_classes = [
        {
            "classe": str(decode_prediction(int(i), label_encoder)),
            "proba": float(preds[i])
        }
        for i in top_idx
    ]
    return label, top_classes


# st.title("🧪 Démonstration des modèles")

# st.markdown("## Vérification des artefacts")
# expected_files = [
#     IMAGE_MODEL_FILE,
#     VECTORIZER_FILE,
#     LABEL_ENCODER_FILE,
#     *TEXT_MODEL_FILES.values(),
# ]
# cols = st.columns(3)
# for i, file_name in enumerate(expected_files):
#     path = MODELS_DIR / file_name
#     with cols[i % 3]:
#         if path.exists():
#             st.success(f"✔ {file_name}")
#         else:
#             st.error(f"✘ {file_name}")

vectorizer = None
label_encoder = None
if (MODELS_DIR / VECTORIZER_FILE).exists() and (MODELS_DIR / LABEL_ENCODER_FILE).exists():
    try:
        vectorizer = load_joblib_artifact(MODELS_DIR / VECTORIZER_FILE)
        label_encoder = load_joblib_artifact(MODELS_DIR / LABEL_ENCODER_FILE)
    except Exception as e:
        st.warning(f"Chargement du vectorizer / label encoder impossible : {e}")

st.markdown("## Démo interactive")
mode = st.radio("Choisir une modalité", ["Texte", "Image", "Comparaison"], horizontal=True)

if mode == "Texte":
    st.subheader("Prédiction texte")
    designation = st.text_input("Désignation", placeholder="Ex. Peluche Pokémon Pikachu 30 cm")
    description = st.text_area("Description", placeholder="Ex. Peluche douce et lavable pour enfant")
    selected_model = st.selectbox("Choisir un modèle texte", list(TEXT_MODEL_FILES.keys()))
    use_stemming = st.toggle("Activer le stemming français", value=False)

    if st.button("Lancer la prédiction texte", use_container_width=True):
        model_path = MODELS_DIR / TEXT_MODEL_FILES[selected_model]
        if not model_path.exists():
            st.error(f"Modèle introuvable : {model_path.name}")
        elif vectorizer is None or label_encoder is None:
            st.error("Le vectorizer TF-IDF ou le label encoder n'ont pas été chargés.")
        else:
            try:
                model = load_joblib_artifact(model_path)
                processed_text = preprocess_text(designation, description, use_stemming=use_stemming)
                pred_label, top_classes = safe_predict_text(model, vectorizer, label_encoder, processed_text)
                st.success(f"Catégorie prédite : **{pred_label}**")
                st.markdown("### Texte après préprocessing")
                st.code(processed_text, language="text")
                if top_classes:
                    st.markdown("### Top 3 des catégories")
                    for item in top_classes:
                        st.write(f"- {item['classe']} : {item['proba']:.4f}")
                else:
                    st.info("Ce modèle ne fournit pas de probabilités via `predict_proba()`.")
            except Exception as e:
                st.exception(e)

elif mode == "Image":
    st.subheader("Prédiction image")
    uploaded = st.file_uploader("Importer une image produit", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        st.image(uploaded, caption="Image chargée", width=320)

    if st.button("Lancer la prédiction image", use_container_width=True):
        model_path = MODELS_DIR / IMAGE_MODEL_FILE
        if uploaded is None:
            st.error("Merci d'importer une image.")
        elif not model_path.exists():
            st.error(f"Modèle image introuvable : {model_path.name}")
        elif label_encoder is None:
            st.error("Le label encoder n'a pas été chargé.")
        else:
            try:
                model = load_keras_artifact(model_path)
                pil_image, img_array = preprocess_image(uploaded)
                pred_label, top_classes = safe_predict_image(model, label_encoder, img_array)
                st.success(f"Catégorie prédite : **{pred_label}**")
                st.image(pil_image, caption="Image prétraitée (224x224 RGB)", width=224)
                st.markdown("### Top 3 des catégories")
                for item in top_classes:
                    st.write(f"- {item['classe']} : {item['proba']:.4f}")
            except Exception as e:
                st.exception(e)

else:
    st.subheader("Comparaison des modèles texte")
    designation = st.text_input("Désignation comparative", placeholder="Ex. Jeu de cartes Pokémon")
    description = st.text_area("Description comparative", placeholder="Ex. Booster neuf sous blister")
    use_stemming = st.toggle("Activer le stemming pour la comparaison", value=False)

    if st.button("Comparer les modèles", use_container_width=True):
        if vectorizer is None or label_encoder is None:
            st.error("Le vectorizer TF-IDF ou le label encoder n'ont pas été chargés.")
        else:
            processed_text = preprocess_text(designation, description, use_stemming=use_stemming)
            rows = []
            for model_name, file_name in TEXT_MODEL_FILES.items():
                model_path = MODELS_DIR / file_name
                if not model_path.exists():
                    rows.append({"Modèle": model_name, "Prédiction": "Fichier absent", "Top 1 probabilité": "-"})
                    continue
                try:
                    model = load_joblib_artifact(model_path)
                    pred_label, top_classes = safe_predict_text(model, vectorizer, label_encoder, processed_text)
                    top1 = f"{top_classes[0]['proba']:.4f}" if top_classes else "N/A"
                    rows.append({"Modèle": model_name, "Prédiction": str(pred_label), "Top 1 probabilité": top1})
                except Exception as e:
                    rows.append({"Modèle": model_name, "Prédiction": f"Erreur: {e}", "Top 1 probabilité": "-"})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.markdown("### Texte après préprocessing")
            st.code(processed_text, language="text")

st.info(
    "Les scripts du projet montrent une logique de préprocessing commune : nettoyage HTML/URLs/ponctuation/chiffres, suppression des stopwords, concaténation des champs texte, TF-IDF côté texte et redimensionnement 224x224 + preprocess_input côté image. "
    "Le stemming a été exposé comme option car il apparaît dans certains scripts, tandis que d'autres injectent `textnostop` dans TF-IDF."
)
