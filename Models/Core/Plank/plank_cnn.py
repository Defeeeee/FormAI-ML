import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

class PlankDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        return features

class PlankCNN(nn.Module):
    def __init__(self):
        super(PlankCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16, 7)  # Corrected input size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16)  # Corrected flattening
        x = self.fc1(x)
        return x

# Split data into training and validation sets
dataset = PlankDataset('../../../Computer_Vision/plank_features.csv')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = PlankCNN()
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

num_epochs = 10000
for epoch in range(num_epochs):
    # Training
    model.train()
    train_epoch_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()

    # Validation
    model.eval()
    val_epoch_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs = data
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_epoch_loss += loss.item()

    # Print training and validation losses
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_epoch_loss / len(train_dataloader):.4f}, '
          f'Val Loss: {val_epoch_loss / len(val_dataloader):.4f}')

torch.save(model.state_dict(), 'plank_model3.pth')