import unittest
import predictor
import pandas as pd
from unittest.mock import patch
from unittest.mock import MagicMock
from pandas.testing import assert_frame_equal


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.data_url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
        self.model_path = "artifacts/model.xgb"
        self.output_path = "output/results.csv"
        # Mocked model for testing
        self.model = MagicMock()
        
        self.encoded_df = pd.DataFrame([[0, 0.5], [1, 0.3], [2, 0.8]], columns=["Feature1", "Feature2"])
        self.predictions = [0.6, 0.4, 0.9]
        self.df = pd.DataFrame([[0], [1], [2]], columns=["ID"])
        self.output_path = "output/results.csv"

    def test_load_data(self):
        # Mock the pd.read_csv function
        with patch('predictor.pd.read_csv') as mock_read_csv:
            mock_df = MagicMock(spec=pd.DataFrame)
            mock_read_csv.return_value = mock_df
            df = predictor.load_data(self.data_url)
            mock_read_csv.assert_called_once_with(self.data_url)
            self.assertIsInstance(df, pd.DataFrame)
    def test_load_model(self):
        # Mock the xgb.Booster and model.load_model functions
        with patch('predictor.xgb.Booster') as mock_booster:
            mock_model = MagicMock()
            mock_booster.return_value = mock_model

            model = predictor.load_model(self.model_path)

            mock_booster.assert_called_once()
            mock_model.load_model.assert_called_once_with(self.model_path)
            self.assertEqual(model, mock_model)
    def test_preprocess_data(self):
        # Mock the preprocess_data implementation
        def mock_preprocess_data(df):
            # Implement mock preprocessing logic
            target = [0, 1, 0, 1]
            features = ["feature1", "feature2"]
            encoded_df = pd.DataFrame([["data1", 0], ["data2", 1]])
            return target, features, encoded_df

        # Patch the preprocess_data function with the mock implementation
        with patch('predictor.preprocess_data', new=mock_preprocess_data):
            df = pd.DataFrame([["data1"], ["data2"]])

            target, features, encoded_df = predictor.preprocess_data(df)

            self.assertEqual(target, [0, 1, 0, 1])
            self.assertEqual(features, ["feature1", "feature2"])
            assert_frame_equal(encoded_df.reset_index(drop=True), pd.DataFrame([["data1", 0], ["data2", 1]]))
    def test_predict(self):
        dtest_mock = MagicMock()
        self.model.predict.return_value = self.predictions  # Set the return value for predict() method
        xgb_mock = MagicMock(DMatrix=MagicMock(return_value=dtest_mock))

        with patch('predictor.xgb', xgb_mock):
            predictions = predictor.predict(self.model, self.encoded_df)

        xgb_mock.DMatrix.assert_called_once_with(self.encoded_df)
        self.model.predict.assert_called_once_with(dtest_mock)
        self.assertEqual(predictions, self.predictions)

if __name__ == '__main__':
    unittest.main(verbosity=2)
