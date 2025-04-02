import os
import unittest
import tempfile
import pandas as pd
from infrastructure.file_io_manager import FileIOManager

class TestFileIOManager(unittest.TestCase):
    def setUp(self):
        self.io_manager = FileIOManager()
        self.temp_csv = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv")
        self.temp_csv.write("name,value\nA,1\nB,2\n")
        self.temp_csv.close()

    def tearDown(self):
        os.unlink(self.temp_csv.name)

    def test_read_csv(self):
        df = self.io_manager.read_csv(self.temp_csv.name)
        self.assertEqual(len(df), 2)
        self.assertIn("name", df.columns)

if __name__ == '__main__':
    unittest.main()
