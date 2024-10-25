import unittest
import torch
from training.train import Classifier

class TestModel(unittest.TestCase):
    def test_model_output_shape(self):
        model = Classifier()
        input_tensor = torch.randn(1, 4)
        output = model(input_tensor)

        assert output.shape == (1, 3)

if __name__ == "__main__":
    unittest.main()