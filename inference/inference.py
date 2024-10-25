import torch
import pandas as pd
from training.train import Classifier
import time
import logging

def load_model():
    
    model = Classifier()
    model.load_state_dict(torch.load('model.pth'))

    return model

def load_data():
    data_df = pd.read_csv('data/inference.csv')
    inp = data_df.drop('target', axis=1).values

    inp = torch.tensor(inp, dtype=torch.float32)

    return data_df, inp
def run_inference():
    try:
        start_time = time.time()
        # Load the trained model
        model = load_model()
        model.eval()

        # Load the data
        data_df, inp = load_data()

        # Run inference
        with torch.no_grad():
            outputs = model(inp)
            _, predicted = torch.max(outputs, 1)

        # Save the predictions
        data_df['predicted'] = predicted.numpy()
        data_df.to_csv('data/predictions.csv', index=False)

        logging.info(f'Inference completed in {time.time() - start_time:.2f} seconds')

        # Calculate accuracy
        correct = (data_df['target'] == data_df['predicted']).sum()
        accuracy = correct / len(data_df)
        print("accuracy: ", accuracy)
        logging.info(f'Accuracy: {accuracy:.4f}')

    except Exception as e:
        logging.error(f"Error during inference: {e}")

if __name__ == "__main__":
    run_inference()