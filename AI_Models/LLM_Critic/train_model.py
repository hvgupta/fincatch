import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LLM_Simulator
from Generate_Dataset import Generate_Dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn import MSELoss
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATASET_SIZE = 10000
INPUT_LOWER_LIMIT = 1
INPUT_UPPER_LIMIT = 100
OUTPUT_LOWER_LIMIT = 1
OUTPUT_UPPER_LIMIT = 10000

NUM_EPOCHS = 1000

# Generate the dataset
dataset_generator = Generate_Dataset(DATASET_SIZE, INPUT_LOWER_LIMIT, INPUT_UPPER_LIMIT, OUTPUT_LOWER_LIMIT, OUTPUT_UPPER_LIMIT)
X, Y = dataset_generator.getX_Y()
dataset = TensorDataset(X, Y)

# Split the dataset into training and testing sets
train_size = int(DATASET_SIZE*0.8)
test_size = int(DATASET_SIZE*0.2)
trainSet, testSet = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing sets
train_loader = DataLoader(trainSet, batch_size=32, shuffle=True)
test_loader = DataLoader(testSet, batch_size=32, shuffle=False)

model = LLM_Simulator(1, 64, 1).to(device)
loss_fn = MSELoss().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for i in range(NUM_EPOCHS):
    model.train()
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = loss_fn(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {i+1}/{NUM_EPOCHS}, Loss: {loss.item()}')
    
# Testing loop
model.eval()
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        X_batch = X_batch.view(-1, 1)
        Y_batch = Y_batch.view(-1, 1)
        Y_pred = model(X_batch)
        loss = loss_fn(Y_pred, Y_batch)
        print(f'Test Loss: {loss.item()}')

test = torch.tensor([[1], [2], [3]],dtype=torch.float32).to(device)  
print(model(test))