from agents import BaseAgent

class RandomAgent(BaseAgent):

    def __init__(self, env, config):
        super().__init__(env, config)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def sample_action(self):
        return self.action_space.sample()
