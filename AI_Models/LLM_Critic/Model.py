from torch import nn
import torch

class LLM_Simulator(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(LLM_Simulator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)