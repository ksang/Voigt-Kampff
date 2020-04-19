class BaseAgent(object):
    """
    BaseAgent is the base class abstractions, all other agent implementations
    are based on this interface.
    Agent is an actor that decoupled from environment, which means this interface
    only implements agent internal flows, interactions with environment should be
    implemented outside of this.
    """
    def __init__(self, env=None, config=None):
        """
        Agent initialization.

        Args:
            env:        gym environment agent is interacting, this is mainly for
                        gathering information, e.g. observation and action space size.
            config:     configurations of the agent, agent type specific.
        """
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.config = config

    def actions_value(self, state):
        """
        Compute distribution of actions and state value of given observation.

        Args:
            state:      observation from environment.
        Returns:
            actions:    a distribution of actions probabilities.
            value:      value of the state.
        """
        raise NotImplementedError

    def take_action(self, state):
        """
        Take one action according agent's behavior policy of given observation.

        Args:
            state:      observation from environment.
        Returns:
            action:     a single action sampled from agent behavior policy.
        """
        raise NotImplementedError

    def update(self, experience):
        """
        Update agent by given experience

        Args:
            experience: trajectory experiences recorded to update agent, e.g.
                        [(s_0, a_0, r_0, s_1, done), ..., (s_t, a_t, r_t, s_{t+1}, done)]
        """
        raise NotImplementedError
