# Voigt-Kampff

Modular implementation of Reinforcement Learning agents and environments.

### About

Project name **Voigt-Kampff** originating in **Philip K. Dick**'s cyberpunk novel [Do Androids Dream of Electric Sheep?](https://en.wikipedia.org/wiki/Do_Androids_Dream_of_Electric_Sheep%3F). It is a polygraph-like machine used by the LAPD's Blade Runners to assist in the testing of an individual to see whether they are a replicant or not, i.e., AI or human. For more details see [Wiki](https://en.wikipedia.org/wiki/Blade_Runner#Voigt-Kampff_machine).

### Environments

- [Tetris](envs/tetris)

### Agents

- [Random](agents/Random)
- [DQN](agents/DQN)
- [A2C](agents/A2C)

### Run

#### Prerequisites

- This project's Python environment is based on [direnv](https://direnv.net/), follow the instructions to install and hook into bash.
- [Pytorch](https://pytorch.org/) is used as deep learning framework.
- Get all git submodules by command:
    ```
    git submodule update --init --recursive
    ```

#### Run scripts

- The run scripts are located in [tyrell](tyrell) folder. For example, below command will train an A2C agent with tetris environment:
    ```
    python3 tyrell/a2c/run_tetris_a2c.py
    ```

### Development

#### pre-commit hooks

- Install pre-commit: [docs](https://pre-commit.com/#installation)
- Install git hooks:
    ```
    pre-commit install
    ```
