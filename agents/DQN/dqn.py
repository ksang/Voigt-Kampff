from agents.DQN import ReplayBuffer
from agents.DQN import Config
from agents import BaseAgent
from agents.common import Linear
from agents.common import CNN

import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

class DQNAgent(BaseAgent):

    def __init__(self, env, config=Config()):
        super().__init__(env, config)
        self.frame_idx = 0
        if config.model_arch == 'linear':
            self.model = Linear(self.observation_space.shape[0], self.config.hidden_dim, self.action_space.n)
        elif config.model_arch == 'cnn':
            self.model = CNN(self.observation_space.shape, self.action_space.n)
        else:
            print("Unknown model:", config.model)
            sys.exit(1)

        if config.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
        else:
            self.Variable = autograd.Variable

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.losses = []

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.config.batch_size)

        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        with torch.no_grad():
            next_state = self.Variable(torch.FloatTensor(np.float32(next_state)))
        action     = self.Variable(torch.LongTensor(action))
        reward     = self.Variable(torch.FloatTensor(reward))
        done       = self.Variable(torch.FloatTensor(done))

        q_values      = self.model(state)
        next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.config.gamma * next_q_value * (1 - done)

        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        return loss

    def normalize(self, state):
        if self.config.normalize:
            state_max, state_min = np.max(state), np.min(state)
            state = 255.0 * (state - state_min) / (state_max - state_min)
        return state

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = self.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
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
