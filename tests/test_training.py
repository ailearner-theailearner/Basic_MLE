import unittest
import torch
from training.train import Classifier, train_model

class TestTraining(unittest.TestCase):
    def test_train_model(self):
        train_model()

        model = Classifier()
        model.load_state_dict(torch.load('model.pth'))
        model.eval()

        input_tensor = torch.randn(1, 4)
        output = model(input_tensor)
        assert output.shape == (1, 3)

if __name__ == "__main__":
    unittest.main()