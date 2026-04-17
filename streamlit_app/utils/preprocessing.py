from __future__ import annotations

import html
import re
from typing import BinaryIO, Iterable

import numpy as np
from PIL import Image

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    preprocess_input = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
except Exception:
    nltk = None
    stopwords = None
    SnowballStemmer = None

HTML_REPLACEMENTS = {
    "<br>": " ",
    "&amp;": " ",
    "&nbsp;": " ",
    "&lt;": " ",
    "&gt;": " ",
    "&quot;": " ",
    "&#39;": " ",
    "&eacute;": "e",
    "&egrave;": "e",
    "&ecirc;": "e",
}

BASIC_FALLBACK_STOPWORDS = {
    "de", "des", "du", "la", "le", "les", "un", "une", "et", "ou", "a", "au", "aux",
    "en", "pour", "avec", "sans", "sur", "par", "dans", "the", "and", "of", "to", "is",
    "are", "this", "that", "generic", "generique"
}


def _ensure_nltk_stopwords() -> None:
    if nltk is None:
        return
    try:
        stopwords.words("french")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def get_stopwords_set() -> set[str]:
    if stopwords is None:
        return BASIC_FALLBACK_STOPWORDS
    _ensure_nltk_stopwords()
    try:
        stop_fr = set(stopwords.words("french"))
        stop_en = set(stopwords.words("english"))
        return stop_fr.union(stop_en).union({"generique"})
    except Exception:
        return BASIC_FALLBACK_STOPWORDS


def get_stemmer():
    if SnowballStemmer is None:
        return None
    try:
        return SnowballStemmer("french")
    except Exception:
        return None


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    for src, dst in HTML_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def delete_stopwords(text: str, stopwords_set: Iterable[str] | None = None) -> str:
    stopwords_set = set(stopwords_set or get_stopwords_set())
    return " ".join([w for w in text.split() if w not in stopwords_set and len(w) > 1])


def stem_text(text: str, stemmer=None) -> str:
    stemmer = stemmer or get_stemmer()
    if stemmer is None:
        return text
    return " ".join(stemmer.stem(w) for w in text.split())


def merge_text_fields(designation: str | None, description: str | None) -> str:
    clean_designation = clean_text(designation)
    clean_description = clean_text(description)
    return f"{clean_designation} {clean_description}".strip()


def preprocess_text(designation: str | None, description: str | None, *, use_stemming: bool = False) -> str:
    merged = merge_text_fields(designation, description)
    without_stopwords = delete_stopwords(merged)
    if use_stemming:
        return stem_text(without_stopwords)
    return without_stopwords


def preprocess_image(uploaded_file: BinaryIO, size: tuple[int, int] = (224, 224)):
    if preprocess_input is None:
        raise ImportError("TensorFlow EfficientNet preprocess_input est indisponible.")
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(size)
    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return image, img_array
