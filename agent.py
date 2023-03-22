from collections import deque
import numpy as np
import torch
import random
from model import Linear_QNet, QTrainer

MAX = 100000
batch_size = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # controls randomness
        self.gamma = 0.9  # discount rate (smaller than 1)
        self.memory = deque(maxlen=MAX)
        self.model = Linear_QNet(15, 256, 3)  # (number of inputs, hidden size, number of outputs)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, grid, snek, apple, border, blockSize):
        head_pos = (int(snek.snake_head.left / blockSize), int(snek.snake_head.top / blockSize))
        direction = snek.snake_head.rotateAngle
        adj_r = (1, 0)
        adj_l = (-1, 0)
        adj_u = (0, -1)
        adj_d = (0, 1)

        danger_s = False
        danger_r = False
        danger_l = False

        dir_l = False
        dir_r = False
        dir_u = False
        dir_d = False

        food_l = False
        food_r = False
        food_u = False
        food_d = False

        tail_l = False
        tail_r = False
        tail_u = False
        tail_d = False

        if not border.hits(snek.snake_head.centerX, snek.snake_head.centerY):
            if direction == 0:
                dir_u = True

                if grid[head_pos[1] + adj_u[1]][head_pos[0] + adj_u[0]] == 1:
                    danger_s = True
                if grid[head_pos[1] + adj_r[1]][head_pos[0] + adj_r[0]] == 1:
                    danger_r = True
                if grid[head_pos[1] + adj_l[1]][head_pos[0] + adj_l[0]] == 1:
                    danger_l = True
            if direction == 90:
                dir_r = True

                if grid[head_pos[1] + adj_r[1]][head_pos[0] + adj_r[0]] == 1:
                    danger_s = True
                if grid[head_pos[1] + adj_d[1]][head_pos[0] + adj_d[0]] == 1:
                    danger_r = True
                if grid[head_pos[1] + adj_u[1]][head_pos[0] + adj_u[0]] == 1:
                    danger_l = True
            if direction == 180:
                dir_d = True

                if grid[head_pos[1] + adj_d[1]][head_pos[0] + adj_d[0]] == 1:
                    danger_s = True
                if grid[head_pos[1] + adj_l[1]][head_pos[0] + adj_l[0]] == 1:
                    danger_r = True
                if grid[head_pos[1] + adj_r[1]][head_pos[0] + adj_r[0]] == 1:
                    danger_l = True
            if direction == 270:
                dir_l = True

                if grid[head_pos[1] + adj_l[1]][head_pos[0] + adj_l[0]] == 1:
                    danger_s = True
                if grid[head_pos[1] + adj_u[1]][head_pos[0] + adj_u[0]] == 1:
                    danger_r = True
                if grid[head_pos[1] + adj_d[1]][head_pos[0] + adj_d[0]] == 1:
                    danger_l = True

        if apple.apple.centerX < snek.snake_head.centerX:
            food_l = True
        if apple.apple.centerX > snek.snake_head.centerX:
            food_r = True
        if apple.apple.centerY < snek.snake_head.centerY:
            food_u = True
        if apple.apple.centerY > snek.snake_head.centerY:
            food_d = True

        if snek.snake_body[0].centerX < snek.snake_head.centerX:
            tail_l = True
        if snek.snake_body[0].centerX > snek.snake_head.centerX:
            tail_r = True
        if snek.snake_body[0].centerY < snek.snake_head.centerY:
            tail_u = True
        if snek.snake_body[0].centerY > snek.snake_head.centerY:
            tail_d = True

        state = [danger_s, danger_r, danger_l, dir_l, dir_r, dir_u, dir_d, food_l, food_r, food_u, food_d, tail_l, tail_r, tail_u, tail_d]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_lm(self):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size) # returns list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_over = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_over)

    def train_sm(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games

        action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action
