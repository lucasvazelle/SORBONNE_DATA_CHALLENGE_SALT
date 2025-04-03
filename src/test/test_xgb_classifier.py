import unittest
import numpy as np
from model.xgb_classifier import XGBMultiLabelClassifier


class TestXGBMultiLabelClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = XGBMultiLabelClassifier()
        np.random.seed(42)
        self.X_train = np.random.rand(10, 5)
        self.y_train = (np.random.rand(10, 3) > 0.5).astype(int)

    def test_train_predict(self):
        self.classifier.train(self.X_train, self.y_train)
        y_pred = self.classifier.predict(self.X_train)
        self.assertEqual(y_pred.shape, self.y_train.shape)


if __name__ == "__main__":
    unittest.main()
