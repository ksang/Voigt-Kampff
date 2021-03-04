from agents import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

    def take_action(self, state):
        return self.action_space.sample()
