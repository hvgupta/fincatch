from torch import nn
import torch

class LLM_Simulator(nn.Module):
    """
    A neural network model simulating a large language model (LLM).

    This class defines a feedforward neural network architecture designed to 
    process input states and produce outputs. The architecture consists of 
    multiple linear layers with a LeakyReLU activation function.
    """
    def __init__(self, state_dim, hidden_dim, output_dim):
        """
        Initialize the LLM_Simulator class.

        Sets up the neural network architecture with specified dimensions.

        Parameters:
        state_dim (int): The dimension of the input state.
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output space.
        """
        super(LLM_Simulator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
         """
        Forward pass through the network to compute the output.

        Parameters:
        x (torch.Tensor): Input tensor representing the state. The shape should 
                          match the input dimension defined during initialization.

        Returns:
        torch.Tensor: The output produced by the neural network based on the input.
        """
        return self.model(x)
