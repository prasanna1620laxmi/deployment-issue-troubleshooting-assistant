"""train_model.py - train, evaluate and save the classifier.  Run: python train_model.py"""
import json
import logging

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import classifier
import config

SEED = 42


def main():
    config.setup_logging()
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    df = classifier.load_dataset()
    labels = sorted(df["category"].unique())
    y = df["category"].map({name: i for i, name in enumerate(labels)}).to_numpy()
    texts = df["log_text"].to_numpy()

    # 70% train, 15% validation, 15% test (kept separate so the test score is honest)
    x_train, x_rest, y_train, y_rest = train_test_split(texts, y, test_size=0.30, stratify=y, random_state=SEED)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED)
    print(f"Rows: train={len(x_train)} validation={len(x_val)} test={len(x_test)}")

    vectorizer = classifier.make_vectorizer(texts=x_train)  # learn vocabulary from training data only
    vocabulary = vectorizer.get_vocabulary()
    model = classifier.build_model(len(vocabulary), len(labels))
    model.fit(
        classifier.vectorize(vectorizer, x_train), y_train,
        validation_data=(classifier.vectorize(vectorizer, x_val), y_val),
        epochs=60, batch_size=16, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    probs = model.predict(classifier.vectorize(vectorizer, x_test), verbose=0)
    y_pred = np.argmax(probs, axis=1)
    report = classification_report(y_test, y_pred, labels=list(range(len(labels))), target_names=labels,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))

    print("\nPer-category results on the TEST set:")
    print(classification_report(y_test, y_pred, labels=list(range(len(labels))), target_names=labels, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(matrix)

    config.MODELS_DIR.mkdir(exist_ok=True)
    model.save(config.MODEL_PATH)
    config.VOCAB_PATH.write_text(json.dumps(vocabulary), encoding="utf-8")
    config.LABELS_PATH.write_text(json.dumps(labels), encoding="utf-8")
    config.METRICS_PATH.write_text(
        json.dumps({"labels": labels, "report": report, "confusion_matrix": matrix.tolist(),
                    "note": "Measured on synthetic data only; not production accuracy."}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved model to {config.MODEL_PATH}")
    print("NOTE: scores on synthetic, template-based data are optimistic. Real logs will score lower.")


if __name__ == "__main__":
    main()
