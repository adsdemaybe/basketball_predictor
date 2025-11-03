import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Configuration
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CSVDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        # Add your data preprocessing steps here
        # (data flipping, percentage calculations, etc.)
        # ...
        
        # Example preprocessing (implement your actual logic):
        df = df.dropna()
        self.X = df.iloc[:, :-1].values.astype(np.float32)
        self.y = df.iloc[:, -1].values.astype(np.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(path):
    # Create dataset and split
    dataset = CSVDataset(path)
    train_size = int((1 - TEST_SIZE) * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=BATCH_SIZE)
    return train_dl, test_dl

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def train_model(train_dl, model, epochs=EPOCHS):
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_dl, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1} Loss: {running_loss/len(train_dl):.4f}')

def evaluate_model(test_dl, model):
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).cpu().numpy()
            preds = (outputs > 0.5).astype(int)
            predictions.extend(preds.ravel().tolist())
            actuals.extend(targets.numpy().ravel().tolist())
    
    return accuracy_score(actuals, predictions)

# Execution
if __name__ == '__main__':
    path = "/Users/advaithvecham/Studying Stuff/PoC/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv"
    train_dl, test_dl = prepare_data(path)
    print(f"Training samples: {len(train_dl.dataset)}, Test samples: {len(test_dl.dataset)}")
    
    model = MLP(34)  # Match your actual input feature count
    train_model(train_dl, model)
    
    acc = evaluate_model(test_dl, model)
    print(f'Accuracy: {acc:.3f}')