import unittest
from src.networks.lstm_model import create_lstm_model

class TestModel(unittest.TestCase):
    def test_model_structure(self):
        """Test the LSTM model structure."""
        model = create_lstm_model((60, 5))
        self.assertEqual(len(model.layers), 6, "Model should have 6 layers")
        self.assertEqual(model.layers[0].output_shape, (None, 60, 50), "Output shape of first LSTM layer should match")
        
if __name__ == '__main__':
    unittest.main()
