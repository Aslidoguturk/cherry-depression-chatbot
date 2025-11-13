import pytest
import numpy as np
import pickle
import json


from chtr3 import clean_text, custom_pad_sequences, predict_with_inputs
from nltk.sentiment import SentimentIntensityAnalyzer

# Load tokenizer
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

vader = SentimentIntensityAnalyzer()
max_length = 300

# --------------------------
# 🚧 Test: Edge Case Inputs
# --------------------------
@pytest.mark.parametrize("text_input", [
    "",                       # Empty string
    "😢😢😢",                 # Emoji-only
    "hello" * 100             # Very long input
])
def test_edge_case_inputs(text_input):
    cleaned = clean_text(text_input)
    sentiment = vader.polarity_scores(cleaned)["compound"]
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = custom_pad_sequences(seq, max_length)
    dummy_phq = np.zeros((1, 8))
    prediction = predict_with_inputs(padded, np.array([[sentiment]]), dummy_phq)
    assert 0 <= prediction <= 1, "Prediction should be between 0 and 1"

# --------------------------
# 💾 Test: Saving Chat Log
# --------------------------
def test_chat_log_saving(tmp_path):
    # Sample chat log
    chat_log = [("user", "I feel sad"), ("cherry", "I'm here for you!")]
    file_path = tmp_path / "test_chat_log.json"

    # Save chat log
    with open(file_path, "w") as f:
        json.dump(chat_log, f)

    # Load and verify
    with open(file_path, "r") as f:
        loaded = json.load(f)

    # Convert back to tuples for comparison
    assert [tuple(item) for item in loaded] == chat_log, "Loaded chat log does not match saved version"


