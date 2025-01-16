from torch import nn

class Policy(nn.Module):
    def __init__(self, state_dim, output_dim, lowerLimit, upperLimit):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, output_dim),
            nn.Sigmoid()
        )
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit

    def forward(self, x):
        return self.model(x)*(self.upperLimit-self.lowerLimit)+self.lowerLimit
    
    def rewardFunction(self, actor_action, critic_action):
        actor_map = {i: 0 for i in range(10)}
        critic_map = {i: 0 for i in range(10)}
        
        for digit in range(4):
            actor_map[actor_action//10**digit%10] += 1
        
        for digit in range(3):
            critic_map[critic_action//10**digit%10] += 1
        same_digit = 0
        for digit in range(10):
            same_digit += min(actor_map[digit], critic_map[digit])
        
        if same_digit == 3: return 100
        if same_digit == 2: return 20
        if same_digit == 1: return 10