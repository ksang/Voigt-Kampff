# Tetris

### About

This work aims to build a modern Tetris game that can interact with Reinforcement Learning agents. It can be also played by human and supports features such as **hard** **drop**, **hold** **queue** and **T-spin**.

To get this env, pull git submodule:
```
git submodule update --init --recursive
```

### States

Two np array:
- main board: (10, 20)
- next queue: (5, 4, 4), 5 Tetromino

### Actions

Discrete, 8 actions:
- 0 : noop
- 1 : move left
- 2 : move right
- 3 : move down
- 4 : hard drop
- 5 : rotate counter-clockwise
- 6 : rotate clockwise
- 7 : hold/dequeue
### Agent play with Gym RL environment
```
import tetris
env = tetris.Tetris()
(initial_board, initial_next_queue) = env.reset()
(board, next_queue), score, done = env.step(env.action_space.sample())
```
