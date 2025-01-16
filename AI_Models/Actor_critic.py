from AI_Models.Actor.Environment import Policy
from AI_Models.LLM_Critic.Model import LLM_Simulator
import torch
import numpy as np


class PPO():
    def __init__(self, Actor: Policy, Critic: LLM_Simulator, device, gamma, lmbda, epochs, eps):
        self.actor = Actor
        self.critic = Critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        
    def compute_advantage(self, td_delta):
        ''' Compute advantage using GAE (Generalized Advantage Estimation) '''
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage_list = np.zeros(len(td_delta))
        for i in reversed(range(len(td_delta))):
            advantage_list[i] = td_delta[i] + (self.gamma * self.lmbda * advantage_list[i + 1] if i + 1 < len(td_delta) else 0)
        
        return torch.tensor(advantage_list, dtype=torch.float).to(self.device)
        