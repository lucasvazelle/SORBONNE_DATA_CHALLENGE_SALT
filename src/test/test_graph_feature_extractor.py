import os
import tempfile
import unittest
from domain.graph_feature_extractor import GraphFeatureExtractor


class TestGraphFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = GraphFeatureExtractor()
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".json"
        )
        self.temp_file.write("INST : mov\nINST : jmp\nINST : call\n")
        self.temp_file.write('"1a2b" -> "3c4d"\n"3c4d" -> "5e6f"\n')
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_extract_features(self):
        text, stats = self.extractor.extract_features(self.temp_file.name)
        self.assertIn("mov", text)
        self.assertEqual(stats["nb_tokens"], 3)
        self.assertEqual(stats["nb_edges"], 2)
        self.assertEqual(stats["nb_nodes"], 3)


if __name__ == "__main__":
    unittest.main()
