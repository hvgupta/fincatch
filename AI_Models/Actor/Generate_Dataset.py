import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AI_Models.LLM_Critic.Model import LLM_Simulator
import torch

class Generate_Dataset:
    def __init__(self, CriticPath: str, datasetSize: int, critic_lowerLimit: int, critic_upperLimit: int, device):
        state_dict = torch.load(CriticPath)
        self.Critic = LLM_Simulator(1, 64, 10000).to(device)
        self.Critic.load_state_dict(state_dict)
        self.Critic.eval()
        self.datasetSize = datasetSize
        self.critic_lowerLimit = critic_lowerLimit
        self.critic_upperLimit = critic_upperLimit
    
    def getX(self) -> torch.Tensor:
        criticInput = torch.randint(self.critic_lowerLimit, self.critic_upperLimit + 1, (self.datasetSize, 1), dtype=torch.float32)
        return self.Critic(criticInput)