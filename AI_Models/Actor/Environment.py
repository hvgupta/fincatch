from torch import nn
import torch

class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim ,output_dim):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)
    
    
def simToReward(sim_output):
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