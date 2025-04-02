import unittest
from domain.codebert_utils import CodeBERTUtils

class TestCodeBERTUtils(unittest.TestCase):
    def test_is_valid_dot_file_invalid(self):
        # On crée un contenu invalide temporairement
        self.assertFalse(CodeBERTUtils.is_valid_dot_file("inexistant.txt"))

    def test_extract_labels(self):
        dot_text = 'label = "foo" some other text label = "bar"'
        labels = CodeBERTUtils.extract_labels_from_text(dot_text)
        self.assertEqual(labels, ["foo", "bar"])

if __name__ == '__main__':
    unittest.main()
