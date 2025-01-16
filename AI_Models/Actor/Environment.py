from torch import nn
import torch

class Policy(nn.Module):
      """
    A neural network-based policy for reinforcement learning.

    This class defines a policy network that maps state observations to actions. 
    The policy network consists of several linear layers with ReLU activations, 
    and outputs are scaled to a specified range defined by lower and upper limits.
    """
    def __init__(self, state_dim, hidden_dim ,output_dim):
          """
    Initialize the Policy class.

    This method sets up the neural network architecture for the policy, 
    which maps state observations to action outputs. The policy network 
    consists of several linear layers with ReLU activations, and the 
    final output is scaled to fit within specified lower and upper limits.

    Parameters:
    state_dim (int): The dimension of the input state space, representing 
                     the number of features in the state observation.
    output_dim (int): The dimension of the output action space, indicating 
                      the number of actions the policy can take.
    lowerLimit (float): The minimum value for the action outputs, used for 
                        scaling the output of the neural network.
    upperLimit (float): The maximum value for the action outputs, used for 
                        scaling the output of the neural network.
    """
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
         """
    Compute the action based on the input state.

    This method performs a forward pass through the policy network, 
    calculating the action to be taken based on the given state input. 
    The output is scaled to fit within the specified lower and upper limits.

    Parameters:
    x (torch.Tensor): Input tensor representing the state. The shape should 
                      match the input dimension defined during initialization.

    Returns:
    torch.Tensor: Scaled action output based on the input state, with values 
                  adjusted to be within the range defined by lowerLimit and 
                  upperLimit.
    """
        return self.model(x)
    
    
def simToReward(sim_output):
    """
    Returns rewards based on the similarity between actor, action and LLM's state.
    
    """
    # Create a new tensor to store the modified values
    modified_output = sim_output[0].clone()
    
    for i in range(len(sim_output)):
        if modified_output[i].item() == 1 or modified_output[i].item() == 2:
            modified_output[i] *= 10
        elif modified_output[i].item() == 3:
            modified_output[i] = modified_output[i].float() * (100 / 3)
        else:
            modified_output[i] *= 0
            
    return modified_output


def rewardFunc(actor_actions, critic_actions, device):
     """
    Compute the reward based on the similarity of actions taken by the actor 
    and the critic.

    This method evaluates the actions of the actor and the critic by counting 
    the number of matching digits in their respective actions. The reward is 
    determined based on how many digits match.

    Parameters:
    actor_action (int): The action taken by the actor, expected to be a 
                        four-digit integer.
    critic_action (int): The action taken by the critic, also expected to be 
                         a four-digit integer.

    Returns:
    int: Reward value based on the number of matching digits:
         - 100 for 3 matches
         - 20 for 2 matches
         - 10 for 1 match
         - 0 for no matches
    """
    # Expand actor and critic actions to digit counts
    same = ((actor_actions//1000)%10 == (critic_actions//1000)%10)*1 + \
            ((actor_actions//100)%10 == (critic_actions//100)%10)*1 + \
            ((actor_actions//10)%10 == (critic_actions//10)%10)*1 + \
            (actor_actions%10 == critic_actions%10)*1
    return simToReward(same), same

class Environment(nn.Module):
    def __init__(self):
        super(Environment, self).__init__()

    def forward(self, actor_actions, critic_actions, device):
        rewards,match = rewardFunc(actor_actions, critic_actions, device)
        return rewards.to(torch.float32).mean(), match
