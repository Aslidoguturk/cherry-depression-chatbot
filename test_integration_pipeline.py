import unittest
import numpy as np
from chtr3 import clean_text, custom_pad_sequences, predict_with_inputs
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle


class TestIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        with open("tokenizer.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        self.max_length = 300
        self.vader = SentimentIntensityAnalyzer()

    def test_end_to_end_text_prediction(self):
        text = "Lately, I have been feeling really down and hopeless."
        cleaned = clean_text(text)
        sentiment = self.vader.polarity_scores(cleaned)["compound"]
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = custom_pad_sequences(seq, self.max_length)
        dummy_phq = np.zeros((1, 8))
        pred = predict_with_inputs(padded, np.array([[sentiment]]), dummy_phq)
        self.assertTrue(0 <= pred <= 1)


if __name__ == "__main__":
    unittest.main()
