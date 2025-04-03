import unittest
from model.codebert_embedder import CodeBERTEmbedder


class TestCodeBERTEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = CodeBERTEmbedder()

    def test_embed_batch(self):
        texts = ["print('Hello, world!')", "def foo(): return 42"]
        embeddings = self.embedder.embed_batch(texts)
        self.assertEqual(embeddings.size(0), len(texts))
        self.assertGreater(embeddings.size(1), 0)


if __name__ == "__main__":
    unittest.main()
