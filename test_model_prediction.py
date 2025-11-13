import unittest
import numpy as np
import pickle
import tensorflow as tf
from chtr3 import clean_text, custom_pad_sequences, predict_with_inputs


class TestModelPrediction(unittest.TestCase):
    def setUp(self):
        with open("tokenizer.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        self.model = tf.keras.models.load_model("lstm_depression_model_with_phq.keras")
        self.max_length = 300

    def test_predict_with_dummy_input(self):
        sample_text = "I feel very low today."
        cleaned = clean_text(sample_text)
        sentiment = 0.0
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = custom_pad_sequences(seq, self.max_length)
        dummy_phq = np.zeros((1, 8))
        prediction = predict_with_inputs(padded, np.array([[sentiment]]), dummy_phq)
        self.assertTrue(0 <= prediction <= 1)


if __name__ == "__main__":
    unittest.main()
