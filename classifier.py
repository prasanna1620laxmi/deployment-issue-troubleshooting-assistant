"""classifier.py - TensorFlow/Keras text classifier for deployment errors.

Design note: text vectorization is done OUTSIDE the saved model (vocabulary in vocab.json),
so the .keras file is a plain model that takes integer sequences. This avoids
version-specific problems with saving string layers.
"""
import json
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

MAX_TOKENS = 5000
SEQUENCE_LENGTH = 40
_LOADED = None


class ModelNotTrainedError(Exception):
    """Raised when predict() is called before train_model.py has been run."""


def load_dataset(path=config.SAMPLE_LOGS_CSV) -> pd.DataFrame:
    """Load and validate the labeled dataset (columns: log_text, category)."""
    df = pd.read_csv(path)
    if not {"log_text", "category"}.issubset(df.columns):
        raise ValueError("Dataset must have the columns: log_text, category")
    df = df.dropna(subset=["log_text", "category"])
    df["log_text"] = df["log_text"].astype(str).str.strip()
    df = df[df["log_text"] != ""].drop_duplicates(subset=["log_text"])
    unknown = set(df["category"]) - set(config.CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories in dataset: {unknown}")
    counts = df["category"].value_counts()
    if len(counts) < len(config.CATEGORIES) or counts.min() < 10:
        raise ValueError("Each category needs at least 10 examples.\n" + counts.to_string())
    return df.reset_index(drop=True)


def make_vectorizer(vocabulary=None, texts=None):
    """Create a Keras TextVectorization layer from a saved vocabulary or by adapting to texts."""
    from tensorflow import keras

    layer = keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_mode="int", output_sequence_length=SEQUENCE_LENGTH
    )
    if vocabulary is not None:
        layer.set_vocabulary(vocabulary)
    else:
        layer.adapt(np.array(texts))
    return layer


def vectorize(vectorizer, texts):
    import tensorflow as tf

    return vectorizer(tf.constant(list(texts))).numpy()


def build_model(vocab_size: int, num_classes: int):
    from tensorflow import keras

    inputs = keras.Input(shape=(SEQUENCE_LENGTH,), dtype="int32")
    x = keras.layers.Embedding(vocab_size, 32)(inputs)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def interpret_probabilities(probs, labels, threshold=config.CONFIDENCE_THRESHOLD) -> dict:
    """Turn raw probabilities into a result dict. Low confidence becomes 'Unknown'."""
    probs = np.asarray(probs, dtype=float)
    best = int(np.argmax(probs))
    confidence = float(probs[best])
    low = confidence < threshold
    return {
        "category": config.UNKNOWN_CATEGORY if low else labels[best],
        "best_guess": labels[best],
        "confidence": round(confidence, 3),
        "low_confidence": low,
        "probabilities": {labels[i]: round(float(p), 3) for i, p in enumerate(probs)},
    }


def _load():
    global _LOADED
    if _LOADED is None:
        if not (config.MODEL_PATH.exists() and config.VOCAB_PATH.exists() and config.LABELS_PATH.exists()):
            raise ModelNotTrainedError("No trained model found. Run: python train_model.py")
        from tensorflow import keras

        model = keras.models.load_model(config.MODEL_PATH)
        vocab = json.loads(config.VOCAB_PATH.read_text(encoding="utf-8"))
        labels = json.loads(config.LABELS_PATH.read_text(encoding="utf-8"))
        _LOADED = (model, make_vectorizer(vocabulary=vocab), labels)
    return _LOADED


def predict(text: str) -> dict:
    """Classify one log message. Returns the dict described in interpret_probabilities()."""
    text = (text or "").strip()
    if not text:
        return {"category": config.UNKNOWN_CATEGORY, "best_guess": None, "confidence": 0.0,
                "low_confidence": True, "probabilities": {}}
    model, vectorizer, labels = _load()
    probs = model.predict(vectorize(vectorizer, [text]), verbose=0)[0]
    return interpret_probabilities(probs, labels)
