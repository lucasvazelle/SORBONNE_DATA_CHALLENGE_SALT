import unittest
from model.text_vectorizer import TextVectorizer
import numpy as np


class TestTextVectorizer(unittest.TestCase):
    def setUp(self):
        self.vectorizer = TextVectorizer(n_components=10)

    def test_fit_transform(self):
        texts = ["this is a test", "another test"]
        X = self.vectorizer.fit_transform(texts)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X.shape[1], 10)

    def test_transform(self):
        texts = ["this is a test"]
        self.vectorizer.fit_transform(texts)
        X = self.vectorizer.transform(texts)
        self.assertEqual(X.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
