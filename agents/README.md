# Agents

Abstraction interfaces and modular implementation of Reinforcement Learning methods.

### Interface

```
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
            config:     configurations of the agent, e.g. learning rate, details see
                        BaseConfig class.
        """

    def actions_value(self, state):
        """
        Compute distribution of actions and state value of given observation.

        Args:
            state:      observation from environment.
        Returns:
            actions:    a distribution of actions probabilities.
            value:      value of the state.
        """

    def sample_action(self, state):
        """
        Sample one action according agent's behavior policy of given observation.

        Args:
            state:      observation from environment.
        Returns:
            action:     a single action sampled from agent behavior policy.
        """

    def update(self, experience):
        """
        Update agent by given experience

        Args:
            experience: trajectory experiences recorded to update agent, e.g.
                        [(s_0, a_0, r_0), ..., (s_t, a_t, r_t)]
        """
```
