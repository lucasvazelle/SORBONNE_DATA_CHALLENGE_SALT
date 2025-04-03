import os
import tempfile
import unittest
from usecases.embedding_pipeline import EmbeddingPipeline
import pandas as pd


class TestEmbeddingPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.save_dir = tempfile.TemporaryDirectory()
        # Créez un fichier JSON factice
        content = 'INST : mov\n"1a2b" -> "3c4d"\n'
        with open(os.path.join(self.test_dir.name, "dummy.json"), "w") as f:
            f.write(content)
        self.pipeline = EmbeddingPipeline(
            json_folder=self.test_dir.name, save_dir=self.save_dir.name, batch_size=1
        )

    def tearDown(self):
        self.test_dir.cleanup()
        self.save_dir.cleanup()

    def test_encode_graphs_in_size_range(self):
        self.pipeline.encode_graphs_in_size_range(min_kb=0, max_kb=1000)
        pt_files = [f for f in os.listdir(self.save_dir.name) if f.endswith(".pt")]
        self.assertGreaterEqual(len(pt_files), 1)


if __name__ == "__main__":
    unittest.main()
