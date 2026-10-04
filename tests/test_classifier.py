import numpy as np
import pytest

import classifier
import config


def test_dataset_loads_and_is_balanced():
    df = classifier.load_dataset()
    assert set(df["category"]) == set(config.CATEGORIES)
    assert df["category"].value_counts().min() >= 10


def test_low_confidence_becomes_unknown():
    result = classifier.interpret_probabilities([0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05], config.CATEGORIES)
    assert result["category"] == config.UNKNOWN_CATEGORY and result["low_confidence"]


def test_high_confidence_keeps_label():
    probs = np.zeros(8)
    probs[3] = 0.95
    probs[0] = 0.05
    assert classifier.interpret_probabilities(probs, config.CATEGORIES)["category"] == config.CATEGORIES[3]


def test_empty_text_is_unknown():
    assert classifier.predict("   ")["category"] == config.UNKNOWN_CATEGORY


def test_trained_model_predicts_known_error():
    pytest.importorskip("tensorflow")
    if not config.MODEL_PATH.exists():
        pytest.skip("Model not trained yet - run python train_model.py first")
    result = classifier.predict("java.lang.OutOfMemoryError: Java heap space")
    assert result["category"] == "Resource Exhaustion"
