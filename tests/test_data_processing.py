import unittest
import pandas as pd
from data_processing import load_and_split_data

class TestDataProcessing(unittest.TestCase):

    def test_load_and_split_data(self):
        load_and_split_data()
        train_df = pd.read_csv('data/training.csv')
        inference_df = pd.read_csv('data/inference.csv')

        assert len(train_df) + len(inference_df) == 150
        assert 'target' in train_df.columns == True
        assert'target' in inference_df.columns == True

if __name__ == "__main__":
    unittest.main()