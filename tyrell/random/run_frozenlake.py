import gym
import argparse
import agents

cmd_parser = argparse.ArgumentParser(description=None)
cmd_parser.add_argument("-t", "--horizon", default=50, help="Maximum time steps")


def render(env, t):
    print("Step:", t)
    env.render()


def run(env, agent, horizon):
    env.reset()
    i = 0
    render(env, i)
    for i in range(horizon):
        _, _, done, _ = env.step(agent.take_action(None))
        if done:
            break
        i += 1
    render(env, i)


if __name__ == "__main__":
    args = cmd_parser.parse_args()
    env = gym.make("FrozenLake-v0")
    agent = agents.Random.RandomAgent(env, None)
    run(env, agent, args.horizon)
