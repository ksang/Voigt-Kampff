from agents.DQN import ReplayBuffer
from agents.DQN import Config
from agents import BaseAgent

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class LinearDQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearDQN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class DQNAgent(BaseAgent):

    def __init__(self, env, config=Config()):
        super().__init__(env, config)
        self.frame_idx = 0
        if config.model_arch == 'linear':
            self.model = LinearDQN(self.observation_space.shape[0], self.config.hidden_dim, self.action_space.n)
        elif config.model_arch == 'cnn':
            self.model = CnnDQN(self.observation_space.shape, self.action_space.n)
        else:
            print("Unknown model:", config.model)
            sys.exit(1)

        if config.use_cuda:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.losses = []

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.config.batch_size)

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))

        q_values      = self.model(state)
        next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.config.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        return loss

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.model.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.action_space.n)
        return action

    def epsilon_by_frame(self, frame_idx):
        c = self.config
        return c.epsilon_final + (c.epsilon_start - c.epsilon_final) * math.exp(-1. * frame_idx/c.epsilon_decay)

    def take_action(self, state):
        self.frame_idx += 1
        return self.act(state, self.frame_idx)

    def update(self, experience):
        # experience: (state, next_state, action, reward, done)
        state, next_state, action, reward, done = experience
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) > self.config.batch_size:
            self.compute_td_loss()
