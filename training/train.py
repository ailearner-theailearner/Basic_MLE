import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time
import logging

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(4, 10)
        self.layer2 = nn.Linear(10, 6)
        self.layer3 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

def get_data_loader():

    #load the data
    train_df = pd.read_csv('data/training.csv')
    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target'].values

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    return train_loader

def train_model():
    try:
        start_time = time.time()
        train_loader = get_data_loader()
        model = Classifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        for epoch in range(100):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

        # Save the trained model
        torch.save(model.state_dict(), 'model.pth')
        logging.info(f'Training completed in {time.time() - start_time:.2f} seconds')
    except Exception as e:
        logging.error(f"Error during training: {e}")

if __name__ == "__main__":
    train_model()