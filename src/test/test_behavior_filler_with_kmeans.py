# tests/test_behavior_filler.py
import unittest
import numpy as np
import pandas as pd
from model.behavior_filler import BehaviorFiller


class TestBehaviorFiller(unittest.TestCase):
    def setUp(self):
        # Création d'un DataFrame fictif pour l'entraînement
        data = {
            "id": ["soft1", "soft2", "soft3"],
            "behav1": [1, 0, 1],
            "behav2": [0, 1, 1],
            "behav3": [1, 1, 0],
        }
        self.metadata_train = pd.DataFrame(data)
        # Création d'un exemple de comportement avec des valeurs manquantes (-1)
        self.known_behaviors = [-1, 1, -1]
        self.filler = BehaviorFiller(n_clusters=2)

    def test_train_kmeans(self):
        kmeans, means = self.filler.train_kmeans(self.metadata_train)
        self.assertIsNotNone(kmeans)
        self.assertIsNotNone(means)

    def test_fill_missing_behaviors(self):
        self.filler.train_kmeans(self.metadata_train)
        filled, cluster = self.filler.fill_missing_behaviors(self.known_behaviors)
        self.assertNotIn(-1, filled)
        self.assertTrue(isinstance(cluster, int))


if __name__ == "__main__":
    unittest.main()
