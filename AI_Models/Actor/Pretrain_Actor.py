from Environment import Policy, Environment
import numpy as np

from Generate_Dataset import Generate_Dataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

DATASET_SIZE = 10000
INPUT_LOWER_LIMIT = 1
INPUT_UPPER_LIMIT = 100

NUM_EPOCHS = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_obj = Generate_Dataset("AI_Models/LLM_Critic/pretrainedCritic.pt", DATASET_SIZE, INPUT_LOWER_LIMIT, INPUT_UPPER_LIMIT, device)
dataset = dataset_obj.getX().to(device).to(torch.float32)
model = Policy(1,64,1000).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = Environment()

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

gamma = 0.8
for epoch in range(NUM_EPOCHS):
    model.train()
    G = []
    Action = []
    Reward = []
    log_output = []
    for i, data in enumerate(dataloader):
        data.requires_grad_(True)
        output = model(data)
        action = torch.argmax(output, dim=1)
        reward,_ = loss_fn(data, action, device)
        G.append(gamma**i * reward)
        Action.append((output, action))
        Reward.append(reward)
        log_output.append(torch.log(output))
    
    G = torch.tensor(G).flip(dims=(0,))
    G = (G - G.mean()) / (G.std() + 1e-9)
    loss = 0
    for g, log_out in zip(G, log_output):
        loss += -g * log_out.gather(1, action.unsqueeze(1)).squeeze(1)
    optimizer.zero_grad()
    loss = loss.sum()
    loss.backward()
    optimizer.step()
    # print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
model.eval()
print(model(torch.tensor([[1109],[678],[8921],[3333],[2367],[89]]).to(device).to(torch.float32)))
torch.save(model.state_dict(), 'AI_Models/Actor/pretrainedActor.pt')