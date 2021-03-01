from agents.A2C import Config
from agents import BaseAgent

import sys
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

class A2CAgent(BaseAgent):

    def __init__(self, env, config=Config()):
        super().__init__(env, config)
        self.frame_idx = 0
        self.losses = []
        self.model = ActorCritic(env.observation_space.shape[0],
                                 env.action_space.n,
                                 config.hidden_dim,
                                 config.std)
        if config.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
            self.device = torch.device("cuda")
        else:
            self.Variable = autograd.Variable
            self.device = torch.device("cpu")

        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def take_action(self, state):
        self.frame_idx += 1
        state = torch.FloatTensor(state).to(self.device)
        dist, value = self.model(state)
        return dist.sample()

    def actions_value(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.model(state)

    def update(self, data):
        '''
        For a fixed number of steps, perform update to the model.
        data = (next_state, log_probs, rewards, masks, values, entropy)
        '''
        (next_state, log_probs, rewards, masks, values, entropy) = data
        next_state = torch.FloatTensor(next_state).to(self.device)
        _, next_value = self.model(next_state)
        returns = self.compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs).detach()
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values).detach()

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.detach().pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        self.losses += [loss]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])
