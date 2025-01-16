from torch import nn

class Policy(nn.Module):
    """
    A neural network-based policy for reinforcement learning.

    This class defines a policy network that maps state observations to actions. 
    The policy network consists of several linear layers with ReLU activations, 
    and outputs are scaled to a specified range defined by lower and upper limits.
    """
    def __init__(self, state_dim, output_dim, lowerLimit, upperLimit):
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
        return self.model(x)*(self.upperLimit-self.lowerLimit)+self.lowerLimit
    
    def rewardFunction(self, actor_action, critic_action):
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
        actor_map = {i: 0 for i in range(10)}
        critic_map = {i: 0 for i in range(10)}
        
        for digit in range(4):
            actor_map[actor_action//10**digit%10] += 1
            critic_map[critic_action//10**digit%10] += 1

        same_digit = 0
        for digit in range(10):
            same_digit += min(actor_map[digit], critic_map[digit])
        
        if same_digit == 3: return 100
        if same_digit == 2: return 20
        if same_digit == 1: return 10
        
        return 0
