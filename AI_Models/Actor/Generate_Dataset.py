import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
import torch


class Generate_Dataset:
    def __init__(self, CriticPath:str, datasetSize:int, critic_lowerLimit:int, critic_upperLimit:int):
        self.Critic = torch.load(CriticPath)
        self.datasetSize = datasetSize
        self.critic_lowerLimit = critic_lowerLimit
        self.critic_upperLimit = critic_upperLimit
    
    def getX(self) -> torch.Tensor:
        criticInput = torch.randint(self.critic_lowerLimit, self.critic_upperLimit+1, (self.datasetSize, 1), dtype=torch.float32)
        return self.Critic(criticInput)
        
        