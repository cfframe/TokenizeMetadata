from pathlib import Path
import os
import pandas as pd
import unittest
from src.meta_data import MetaData


class MetaDataTestCase(unittest.TestCase):
    def setUp(self):
        """Fixtures used by test."""
        self.Root = Path(__file__).parent
        self.TestDataDir = os.path.join(self.Root, 'test_data')
        self.TsvFilePath = os.path.join(self.TestDataDir, 'test_tsv_5_cols_inc_labels.txt')

    def test_read_raw_data__raw_data_has_expected_shape(self):
        md = MetaData(self.TsvFilePath)

        pd.set_option('display.max_colwidth', 100)
        print(md.raw_data.head())
        self.assertEqual(md.raw_data.shape, (7, 5))


if __name__ == '__main__':
    unittest.main()
