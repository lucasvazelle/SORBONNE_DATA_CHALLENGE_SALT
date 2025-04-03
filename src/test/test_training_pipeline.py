import os
import tempfile
import unittest
import pandas as pd
from unittest.mock import patch
from usecases.training_pipeline import TrainingPipeline


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        # Création d'un CSV temporaire pour les métadonnées
        self.train_csv_fd, self.train_csv_path = tempfile.mkstemp(suffix=".csv")
        df_train = pd.DataFrame({"name": ["dummy"], "target": [1]})
        df_train.to_csv(self.train_csv_path, index=False, sep=";")
        # Répertoire temporaire pour les graphes
        self.train_graph_dir = tempfile.TemporaryDirectory()
        dummy_content = 'INST : mov\n"node1" -> "node2"\n'
        with open(os.path.join(self.train_graph_dir.name, "dummy.json"), "w") as f:
            f.write(dummy_content)
        # Répertoires pour embeddings et sortie
        self.embedding_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        os.close(self.train_csv_fd)
        os.remove(self.train_csv_path)
        self.train_graph_dir.cleanup()
        self.embedding_dir.cleanup()
        self.output_dir.cleanup()

    def test_run_pipeline(self):
        with patch("pandas.DataFrame.sample", lambda self, n, random_state: self):
            pipeline = TrainingPipeline(
                train_csv=self.train_csv_path,
                train_graph_dir=self.train_graph_dir.name,
                embedding_dir=self.embedding_dir.name,
                output_dir=self.output_dir.name,
                sample_size=1,
            )
            pipeline.run()
            model_path = os.path.join(self.output_dir.name, "xgb_model_full.pkl")
            self.assertTrue(os.path.exists(model_path))


if __name__ == "__main__":
    unittest.main()
