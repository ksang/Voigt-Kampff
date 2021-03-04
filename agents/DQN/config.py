class Config(object):
    def __init__(self):
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 100000
        # dimentaion of hidden layer for DQN
        self.hidden_dim = 128
        self.use_cuda = True
        self.replay_buffer_size = 1000
        self.batch_size = 32
        self.model_arch = "linear"
        self.gamma = 0.99
        self.normalize = True
