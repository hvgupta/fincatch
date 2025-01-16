import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Critic.Model import LLM_Simulator
import torch

class Generate_Dataset:
     """
     A class to generate datasets using a pre-trained Critic model.

     This class initializes the Critic model from a specified path and 
     generates input data for the model based on defined limits and 
     dataset size. It provides methods to create input tensors and 
     obtain model outputs.
     """
     def __init__(self, CriticPath: str, datasetSize: int, critic_lowerLimit: int, critic_upperLimit: int, device):
          """
          Initialize the Generate_Dataset class.

          This method sets up the Critic model by loading its state from a specified 
          path and prepares parameters for generating a dataset. The Critic model 
          is initialized and set to evaluation mode.

          Parameters:
          CriticPath (str): The file path to the saved state dictionary of the 
                              Critic model. This should point to a `.pth` file 
                              containing the model's weights.
          datasetSize (int): The number of input samples to generate for the dataset.
          critic_lowerLimit (int): The minimum value for the random inputs that 
                                   will be passed to the Critic model.
          critic_upperLimit (int): The maximum value for the random inputs that 
                                   will be passed to the Critic model.
          device: The device on which to load the model (e.g., 'cpu' or 'cuda').
          """
          state_dict = torch.load(CriticPath)
          self.Critic = LLM_Simulator(1, 64, 10000).to(device)
          self.Critic.load_state_dict(state_dict)
          self.Critic.eval()
          self.datasetSize = datasetSize
          self.critic_lowerLimit = critic_lowerLimit
          self.critic_upperLimit = critic_upperLimit
          self.device = device

     def getX(self) -> torch.Tensor:
          """
     Generate input tensors for the Critic model and compute the corresponding output.

     This method creates a tensor of random integers within a specified range 
     and feeds it into the Critic model to obtain the output tensor. The 
     generated input values are used as features for the Critic.

     Returns:
     torch.Tensor: The output tensor from the Critic model based on the 
                    randomly generated inputs.
 """
          criticInput = torch.randint(self.critic_lowerLimit, self.critic_upperLimit + 1, (self.datasetSize, 1), dtype=torch.float32).to(self.device)
          return torch.argmax(self.Critic(criticInput),dim=1).view(-1,1)
