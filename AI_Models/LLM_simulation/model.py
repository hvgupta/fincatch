from torch import nn

class LLM_Simulator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lowerLimit, upperLimit):
        super(LLM_Simulator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU()
        )
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit

    def forward(self, x):

        return self.model(x)*(self.upperLimit-self.lowerLimit)+self.lowerLimit