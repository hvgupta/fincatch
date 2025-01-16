from Actor.Environment import Policy, Environment
from LLM_Critic.Model import LLM_Simulator
import torch
import numpy as np


class Actor_Critic():
    """A class that implements the Actor-Critic reinforcement learning algorithm.

    This class initializes and manages the actor and critic networks, 
    loads their pre-trained weights, and contains methods for updating 
    the networks based on interactions with an environment.
    """
    def __init__(self, pretrainedActorPath:str , PreTrainedCriticPath: str ,device, gamma, lmbda, epochs, eps):
        """
        Initialize the Actor-Critic model with given parameters.

        Args:
            pretrainedActorPath (str): Path to the pre-trained actor model.
            PreTrainedCriticPath (str): Path to the pre-trained critic model.
            device: The device to run the model on (CPU or GPU).
            gamma: Discount factor for future rewards.
            lmbda: Lambda parameter for Generalized Advantage Estimation.
            epochs: Number of training epochs.
            eps: Epsilon for exploration.
        """
        self.actor = Policy(1,64,1000).to(device)
        self.critic = LLM_Simulator(1, 64, 10000).to(device)
        self.actor.load_state_dict(torch.load(pretrainedActorPath))
        self.critic.load_state_dict(torch.load(PreTrainedCriticPath))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.envir = Environment()
    
    def end(self,count) -> bool:
         """Check if the training episode should end.
        Args:
            count: The current count of actions taken.
        Returns:
            bool: True if the episode should end, otherwise False.
        """
        if count == 10:
            return True
    
    def update(self, startingInput):
         """
         Update the actor-critic model for a specified number of epochs.

        Args:
            startingInput: The initial input state for the environment.
        """
        for epoch in range(self.epochs):
            G = []
            Action = []
            Reward = []
            log_output = []
            input = startingInput
            count = 0
            while True:
                state = self.critic(input)
                state.requires_grad_(True)
                output = self.actor(state)
                action = torch.argmax(output, dim=1)
                reward, input = self.envir(state, action)
                G.append(self.gamma * reward)
                Action.append((output, action))
                Reward.append(reward)
                log_output.append(torch.log(output))
                count += 1
                if self.end():
                    break
                
            G = torch.tensor(G).flip(dims=(0,))
            G = (G - G.mean()) / (G.std() + 1e-9)
            loss = 0
            for g, log_out in zip(G, log_output):
                loss += -g * log_out.gather(1, action.unsqueeze(1)).squeeze(1)
            self.actor_optimizer.zero_grad()
            loss = loss.sum()
            loss.backward()
            self.actor_optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
