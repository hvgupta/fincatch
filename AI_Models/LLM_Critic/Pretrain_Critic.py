import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AI_Models.LLM_Critic.Model import LLM_Simulator
from Generate_Dataset import Generate_Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
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
X, Y = dataset_generator.generate_dataset_io_tensors()
dataset = TensorDataset(X, Y)

# Create DataLoaders for training and testing sets
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LLM_Simulator(1, 64, 10000).to(device)
loss_fn = CrossEntropyLoss().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for i in range(NUM_EPOCHS):
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = loss_fn(Y_pred, Y_batch.view(-1).long())
        loss.backward()
        optimizer.step()
    print(f'Epoch {i+1}/{NUM_EPOCHS}, Loss: {loss.item()}')

# test = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [13], [56], [89], [100]],dtype=torch.float32).to(device)  
# print(torch.argmax(model(test),dim=1))

torch.save(model.state_dict(), '/AI_Models/LLM_Critic/pretrainedCritic.pt')
