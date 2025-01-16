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
    
import torch

def expand_to_digit_counts(tensor, device):
    # Get the shape of the input tensor and add a new dimension for the 10 digits
    shape = list(tensor.shape) + [10]
    # Create an output tensor filled with zeros
    digit_counts = torch.zeros(*shape, dtype=torch.int64, device=device)
    
    # Flatten the input tensor for easier processing
    flat_tensor = tensor.view(-1)
    # Flatten the output tensor for easier indexing
    flat_digit_counts = digit_counts.view(-1, 10)
    
    for i, value in enumerate(flat_tensor):
        value = abs(value.item())  # Ensure the value is non-negative
        digit_count = 0
        
        # Count digits and populate the digit tensor
        while value > 0:
            digit = value % 10
            flat_digit_counts[i, digit] += 1
            value //= 10
            digit_count += 1
        
        # Add padding for numbers with fewer than 4 digits
        if digit_count < 4:
            flat_digit_counts[i, 0] += (4 - digit_count)
    
    # Reshape back to the original shape with the new dimension
    return digit_counts

def simDigitToReward(digit_counts):
    for i in range(len(digit_counts)):
        if digit_counts[i]== 1 or digit_counts[i]== 2:
            digit_counts[i]*= 10
        elif digit_counts[i]== 3:
            digit_counts[i]*= 100/3
        else:
            digit_counts[i]*= 0
    return digit_counts
        

def lossFunc(actor_actions, critic_actions, device):
    
    actor_digit_counts = expand_to_digit_counts(actor_actions,device)
    critic_digit_counts = expand_to_digit_counts(critic_actions,device)
    
    min_digit_counts = torch.minimum(actor_digit_counts, critic_digit_counts)
    
    return simDigitToReward(min_digit_counts.sum(dim=1))


print(lossFunc(torch.tensor([937, 123, 456, 789]), torch.tensor([89, 1, 444, 789]), 'cpu'))




