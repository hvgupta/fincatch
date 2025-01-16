from torch import nn

class LLM_Simulator(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim, lowerLimit, upperLimit):
        super(LLM_Simulator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit

    def forward(self, x):
        return self.model(x)*(self.upperLimit-self.lowerLimit)+self.lowerLimit