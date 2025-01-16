from Actor.Policy import Policy
from LLM_Critic.model import LLM_Simulator
import torch


class PPO():
    def __init__(self, state_dim, hidden_dim ,output_dim, actor_lowerLimit, actor_upperLimit, critic_lowerLimit, critic_upperLimit, device, gamma, lmbda, epochs, eps):
        self.actor = Policy(state_dim, output_dim, actor_lowerLimit, actor_upperLimit).to(device)
        self.critic = LLM_Simulator(state_dim, hidden_dim, output_dim, critic_lowerLimit, critic_upperLimit).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        