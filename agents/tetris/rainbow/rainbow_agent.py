import math, random
import argparse
import os,sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from layers import NoisyLinear
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
# Hyper-paramenters
NUM_ATOMS               = 51
VMIN                    = -10
VMAX                    = 10
REPLAY_SIZE             = 100000
REPLAY_INITIAL          = 1000
EPSILON_TRAIN           = 0.02
EPSILON_EVAL            = 0.001
EPSILON_DECAY_PERIOD    = 1000

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir,
    os.path.pardir)))

from envs.tetris import Game

cmd_parser = argparse.ArgumentParser(description=None)
# Arguments
cmd_parser.add_argument('-m', '--mode', default='train', choices=['train', 'play'],
                        help='Run agent with train or play mode.')
cmd_parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float,
                        help='Optimizor learning rate.')
cmd_parser.add_argument('-fn', '--frame_num', default=100000, type=int,
                        help='Number of frames for training.')
cmd_parser.add_argument('-bs', '--batch_size', default=32, type=int,
                        help='Batch size for training.')
cmd_parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='Discount value for training.')

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon parameter for the agent's e-greedy policy.
  Args:
    decay_period: float, the decay period for epsilon.
    step: Integer, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before training starts.
    epsilon: float, the epsilon value.
  Returns:
    A float, the linearly decaying epsilon value.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
  return epsilon + bonus

class TetrisRainbowCnnDQN(nn.Module):
    """
    Rainbow CNN DQN crafted for tetris environment.
    Initial paramenters:
        input_shape_main        input shape for main board (width, height, channel)
        input_shape_side        input shape for side board (width, height, channel)
        num_actions             number of total possible actions
        epsilon_fn              function to calculate epsilon
        epsilon_train           float, base epsilon for training
        epsilon_eval            float, epsilon during evaluation
        epsilon_decay_period    int, number of steps for epsilon to decay
        num_atoms               number of atoms used for distibutional RL
        Vmin, Vmax              min and max value used for distibutional RL
        eval_mode               flag indicate skipping training logic

    """
    def __init__(self,
                input_shape_main,
                input_shape_side,
                num_actions,
                epsilon_train=EPSILON_TRAIN,
                epsilon_eval=EPSILON_EVAL,
                epsilon_decay_period=EPSILON_DECAY_PERIOD,
                num_atoms=NUM_ATOMS,
                Vmin=VMIN,
                Vmax=VMAX,
                eval_mode=False):
        super(TetrisRainbowCnnDQN, self).__init__()

        self.input_shape_main       = input_shape_main
        self.input_shape_side       = input_shape_side
        self.num_actions            = num_actions
        self.epsilon_train          = epsilon_train
        self.epsilon_eval           = epsilon_eval
        self.epsilon_decay_period   = epsilon_decay_period
        self.num_atoms              = num_atoms
        self.Vmin                   = Vmin
        self.Vmax                   = Vmax
        self.eval_mode              = eval_mode
        self.training_steps         = 0

        self.features_main = nn.Sequential(
            nn.Conv2d(input_shape_main[0], 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.features_side = nn.Sequential(
            nn.Conv2d(input_shape_side[0], 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.noisy_value1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(512, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        # divide by color space
        x1 = x1 / 10.
        x1 = self.features_main(x1)
        x1 = x1.view(batch_size, -1)
        x2 = x2 / 10.
        x2 = self.features_side(x2)
        x2 = x2.view(batch_size, -1)
        # Concatenate main board and side board features
        x = torch.cat((x1, x2),1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value     = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        # perform softmax on all action dimension
        x = F.softmax(x.view(-1, self.num_atoms), dim=0).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def feature_size(self):
        main = self.features_main(autograd.Variable(torch.zeros(1, *self.input_shape_main))).view(1, -1).size(1)
        side = self.features_side(autograd.Variable(torch.zeros(1, *self.input_shape_side))).view(1, -1).size(1)
        return main+side

    def get_epsilon(self):
        epsilon = linearly_decaying_epsilon(self.epsilon_decay_period,
                                            self.training_steps,
                                            REPLAY_INITIAL,
                                            self.epsilon_train)
        return epsilon

    def act(self, state):
        epsilon = self.get_epsilon()
        self.training_steps += 1
        if random.random() <= epsilon:
          return np.random.randint(self.num_actions)
        with torch.no_grad():
            main = Variable(torch.FloatTensor(np.float32(state[0])).unsqueeze(0)).reshape(1,1,10,20)
            side = Variable(torch.FloatTensor(np.float32(state[1])).unsqueeze(0)).reshape(1,1,-1,4)
            dist = self.forward(main, side).data.cpu()
            dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
            action = dist.sum(2).max(1)[1].numpy()[0]
            return action

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def projection_distribution(target_model, next_main, next_side, rewards, dones, Vmin, Vmax, num_atoms):
    batch_size  = next_main.size(0)

    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)

    next_dist   = target_model(next_main, next_side).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist   = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones   = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b  = (Tz - Vmin) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist

def compute_td_loss(optimizer, current_model, target_model, replay_buffer, batch_size, Vmin, Vmax, num_atoms):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    main = Variable(torch.FloatTensor(np.float32(state[:,0].tolist())).unsqueeze(0)).reshape(batch_size,1,10,20)
    side = Variable(torch.FloatTensor(np.float32(state[:,1].tolist())).unsqueeze(0)).reshape(batch_size,1,-1,4)
    next_main = Variable(torch.FloatTensor(np.float32(next_state[:,0].tolist())).unsqueeze(0)).reshape(batch_size,1,10,20)
    next_side = Variable(torch.FloatTensor(np.float32(next_state[:,1].tolist())).unsqueeze(0)).reshape(batch_size,1,-1,4)
    action     = Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))

    proj_dist = projection_distribution(target_model,
                                        next_main,
                                        next_side,
                                        reward,
                                        done,
                                        Vmin,
                                        Vmax,
                                        num_atoms)

    dist = current_model(main, side)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    loss = -(Variable(proj_dist) * dist.log()).sum(1)
    loss  = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()

    return loss

def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

def train(args):
    env = Game()

    current_model = TetrisRainbowCnnDQN((1,10,20), (1,20,4), env.action_space.n, NUM_ATOMS, VMIN, VMAX)
    target_model  = TetrisRainbowCnnDQN((1,10,20), (1,20,4), env.action_space.n, NUM_ATOMS, VMIN, VMAX)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()

    optimizer = optim.Adam(current_model.parameters(), lr=args.learning_rate)
    update_target(current_model, target_model)

    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    for frame_idx in range(1, args.frame_num + 1):
        action = current_model.act(state)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        #print("state:")
        #print(state)
        #print("action: %d reward: %d done: %d" %(action, reward, done))
        #print("next_state")
        #print(next_state)
        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > REPLAY_INITIAL:
            loss = compute_td_loss( optimizer,
                                    current_model,
                                    target_model,
                                    replay_buffer,
                                    args.batch_size,
                                    Vmin=VMIN,
                                    Vmax=VMAX,
                                    num_atoms=NUM_ATOMS)
            losses.append(loss.data.item())

        if frame_idx % 1000 == 0:
            #print(all_rewards)
            #print(losses)
            plot(frame_idx, all_rewards, losses)

        if frame_idx % 100 == 0:
            update_target(current_model, target_model)

def play(args):
    return

if __name__ == '__main__':
    args = cmd_parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'play':
        play(args)
