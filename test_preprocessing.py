import unittest

from preprocessing_utils import clean_text_advanced, synonym_augment


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_advanced(self):
        text = "I'm Feeling so GREAT!!!"
        result = clean_text_advanced(text)
        self.assertNotIn("!", result)
        self.assertIn("feeling", result.lower())

    def test_synonym_augment(self):
        text = "I feel sad and tired"
        result = synonym_augment(text)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")


if __name__ == "__main__":
    unittest.main()
