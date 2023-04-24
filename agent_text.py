from collections import deque
import numpy as np
import torch
import random
from model import Linear_QNet, QTrainer
import shelve
import os

MAX = 100000
batch_size = 1000
LR = 0.001
RANDOM = 500


def find_dist(grid, head_pos, change_y, change_x):
    dist = 0
    has_apple = False
    space = 0

    while space != 1:
        space = grid[head_pos[0] + (change_y * (dist+1))][head_pos[1] + (change_x * (dist+1))]
        dist += 1
        if space == 9:
            has_apple = True

    return dist, has_apple

class Agent:
    def __init__(self, n_games=0):
        self.n_games = n_games
        self.epsilon = 0  # controls randomness
        self.gamma = 0.9  # discount rate (smaller than 1)
        #self.memory = deque(maxlen=MAX)
        self.model = Linear_QNet(16, 32, 3)  # (number of inputs, hidden size, number of outputs)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        model_folder_path = './model'
        file_path = 'memory.pickle'
        file_name = os.path.join(model_folder_path, file_path)

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

            print('not found')
            self.memory = shelve.open(file_name, writeback=True)
            self.memory['mem'] = deque(maxlen=MAX)

        self.memory = shelve.open(file_name, writeback=True)

    def get_state(self, grid, snek, apple):
        head_pos = list(snek)[0]
        apple_pos = apple

        change = snek.get(list(snek)[0])

        if change == (0, 1):        # right
            direction_head = 90
        elif change == (1, 0):      # down
            direction_head = 180
        elif change == (0, -1):     # left
            direction_head = 270
        elif change == (-1, 0):     # up
            direction_head = 0

        change_tail = snek.get(list(snek)[-1])

        if change_tail == (0, 1):        # right
            direction_tail = 90
        elif change_tail == (1, 0):      # down
            direction_tail = 180
        elif change_tail == (0, -1):     # left
            direction_tail = 270
        elif change_tail == (-1, 0):     # up
            direction_tail = 0

        dist_s = 0
        dist_r = 0
        dist_l = 0

        dist_food_x = 0
        dist_food_y = 0

        food_s = False
        food_r = False
        food_l = False

        dir_l = False
        dir_r = False
        dir_u = False
        dir_d = False

        tail_dir_l = False
        tail_dir_r = False
        tail_dir_u = False
        tail_dir_d = False

        if list(snek)[0][0] not in (0, 19) and list(snek)[0][1] not in (0, 19):
            dist_food_x = abs(head_pos[1] - apple_pos[1])
            dist_food_y = abs(head_pos[0] - apple_pos[0])

            if direction_head == 0:
                dir_u = True

                dist_s, food_s = find_dist(grid, head_pos, -1, 0)
                dist_r, food_r = find_dist(grid, head_pos, 0, 1)
                dist_l, food_l = find_dist(grid, head_pos, 0, -1)

            if direction_head == 90:
                dir_r = True

                dist_s, food_s = find_dist(grid, head_pos, 0, 1)
                dist_r, food_r = find_dist(grid, head_pos, 1, 0)
                dist_l, food_l = find_dist(grid, head_pos, -1, 0)

            if direction_head == 180:
                dir_d = True

                dist_s, food_s = find_dist(grid, head_pos, 1, 0)
                dist_r, food_r = find_dist(grid, head_pos, 0, -1)
                dist_l, food_l = find_dist(grid, head_pos, 0, 1)

            if direction_head == 270:
                dir_l = True

                dist_s, food_s = find_dist(grid, head_pos, 0, -1)
                dist_r, food_r = find_dist(grid, head_pos, -1, 0)
                dist_l, food_l = find_dist(grid, head_pos, 1, 0)

            if direction_tail == 0:
                tail_dir_u = True

            if direction_tail == 90:
                tail_dir_r = True

            if direction_tail == 180:
                tail_dir_d = True

            if direction_tail == 270:
                tail_dir_l = True

        state = [dist_s, dist_r, dist_l, dist_food_y, dist_food_x, food_s, food_r, food_l,
                 dir_l, dir_r, dir_u, dir_d, tail_dir_l, tail_dir_r, tail_dir_u, tail_dir_d]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory['mem'].append((state, action, reward, next_state, game_over))

    def forget(self, steps):
        for i in range(steps):
            self.memory['mem'].pop()

    def train_lm(self):
        if len(self.memory['mem']) > batch_size:
            mini_sample = random.sample(self.memory['mem'], batch_size) # returns list of tuples
        else:
            mini_sample = self.memory['mem']

        states, actions, rewards, next_states, game_over = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_over)

    def train_sm(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state, best_score):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = RANDOM - self.n_games

        action = [0, 0, 0]

        if random.randint(0, RANDOM*2) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action
