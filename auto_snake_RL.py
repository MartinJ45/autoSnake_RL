# Name: Martin Jimenez
# Date: 03/17/2023 (last updated)

from cmu_graphics import *
from snake_classes import Snake
from snake_classes import Apple
import numpy as np
import torch
import random
from collections import deque

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from helper import plot

# The default game speed is 10; this value can be changed by pressing the
# left and right arrow keys
app.stepsPerSecond = 100
step = 0

isPaused = False
isPlaying = False
reset = False
autoReset = True

# Grabs size input
msg = 'Enter the board length (default 18 or press ENTER)'
size = app.getTextInput(msg)
while not size.isdigit() or int(size) % 2 != 0 or int(size) > 200 or int(size) < 4:
    if size == '':
        size = 18
        break
    if not size.isdigit():
        msg = 'Enter the board length (enter a digit)'
    if int(size) % 2 != 0:
        msg = 'Enter the board length (only evens)'
    if int(size) > 200:
        msg = 'Enter the board length (max size 200)'
    if int(size) < 4:
        msg = 'Enter the board length (min size 4)'

    size = app.getTextInput(msg)

size = int(size)
blockSize = 400 / (size + 2)

# This is the grid that the snake uses
# 0 represents open space
# 1 represents the border or the snake body
# 2 represents the path the snake plans to take towards the apple
# 3 represents the snake head
# 5 is a goal representing the end of the tail (used when it cannot locate the apple)
# 9 is a goal representing the apple
grid = [
    [1] * (size + 2)
]

for i in range(size):
    grid.append([1] + [0] * size + [1])
grid.append([1] * (size + 2))

# Draws the background grid
gridBackground = Group()
for x in range(size + 2):
    gridBackground.add(Line(blockSize * x, 0, blockSize * x, 400, lineWidth=1))
for y in range(size + 2):
    gridBackground.add(Line(0, blockSize * y, 400, blockSize * y, lineWidth=1))

# Draws the border and score
border = Polygon(
    0,
    0,
    400,
    0,
    400,
    400,
    0,
    400,
    0,
    blockSize,
    blockSize,
    blockSize,
    blockSize,
    400 - blockSize,
    400 - blockSize,
    400 - blockSize,
    400 - blockSize,
    blockSize,
    0,
    blockSize)
score = Label(0, 50, blockSize / 2, fill='white', size=blockSize)
game_m = Label('Game:', 180, blockSize / 2, fill='white', size=blockSize)
game = Label(0, 250, blockSize / 2, fill='white', size=blockSize)

path = []

appleSeed = []
snek = Snake(200, 200, blockSize, size)
apple = Apple(300, 200, blockSize, size)

reward = 0

gameOverMessage = Label('GAME OVER', 200, 200, size=50, fill='red', visible=False)

MAX = 100000
batch_size = 1000
LR = 0.001


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )  # defines it as a tuple

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for i in range(len(game_over)):
            Q_new = reward[i]

            if not game_over[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_Q) -> only if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # controls randomness
        self.gamma = 0.9  # discount rate (smaller than 1)
        self.memory = deque(maxlen=MAX)
        self.model = Linear_QNet(11, 256, 3)  # (number of inputs, hidden size, number of outputs)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, grid):
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

        state = [danger_s, danger_r, danger_l, dir_l, dir_r, dir_u, dir_d, food_l, food_r, food_u, food_d]

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


plot_scores = []
plot_avg_scores = []
total_score = 0
best_score = 0
agent = Agent()


# Stops the program when the snake loses, wins, or if the game is ended early
def gameOver():
    global reward

    if autoReset:
        print("appleSeed =", apple.get_seed())
        resetGame()

        reward = -10
        return

    if not gameOverMessage.visible:
        # The max value it can achieve is 324 in an 18 by 18 grid
        if score.value == pow(size, 2) - 1:
            gameOverMessage.value = 'YOU WIN!'
            gameOverMessage.fill = 'lime'
        else:
            gameOverMessage.value = 'GAME OVER'
            gameOverMessage.fill = 'red'

        gameOverMessage.visible = True
        gameOverMessage.toFront()

        print("appleSeed =", apple.get_seed())

        reward = -10


def resetGame():
    global appleSeed
    global snek
    global apple
    global step

    print('RESET')
    print('Snake got', score.value)

    appleSeed = []

    snek.reset()
    apple.reset()

    snek = Snake(200, 200, blockSize, size)
    apple = Apple(300, 200, blockSize, size)

    score.value = 0
    step = 0

    gameOverMessage.visible = False


# Updates the grid
def genGrid():
    grid = [
        [1] * (size + 2)
    ]

    for i in range(size):
        grid.append([1] + [0] * size + [1])
    grid.append([1] * (size + 2))

    for body in snek.snake_body:
        grid[int(body.top / blockSize)][int(body.left / blockSize)] = 1

    grid[int(snek.snake_head.top / blockSize)
         ][int(snek.snake_head.left / blockSize)] = 3

    grid[int(apple.apple.top / blockSize)
         ][int(apple.apple.left / blockSize)] = 9

    return grid


def onKeyPress(key):
    global isPaused
    global isPlaying
    global reset
    global autoReset

    if key == 'R':
        reset = True

    if key == 'A':
        if autoReset:
            autoReset = False
            print('Auto reset off')
        else:
            autoReset = True
            print('Auto reset on')

    # Allows the player to play
    if key == 'P':
        if isPlaying:
            isPlaying = False
            print('Computer in control')
        else:
            isPlaying = True
            print('Player in control')
            print('Use WASD to move')

        isPaused = True
        print('Paused the game')

    # Player controls if the player is playing
    if isPlaying:
        if key == 'w':
            if snek.snake_head.rotateAngle == 0:
                snek.set_direction('forward')
            if snek.snake_head.rotateAngle == 90:
                snek.set_direction('turn_left')
            if snek.snake_head.rotateAngle == 270:
                snek.set_direction('turn_right')
        if key == 's':
            if snek.snake_head.rotateAngle == 180:
                snek.set_direction('forward')
            if snek.snake_head.rotateAngle == 90:
                snek.set_direction('turn_right')
            if snek.snake_head.rotateAngle == 270:
                snek.set_direction('turn_left')
        if key == 'a':
            if snek.snake_head.rotateAngle == 0:
                snek.set_direction('turn_left')
            if snek.snake_head.rotateAngle == 180:
                snek.set_direction('turn_right')
            if snek.snake_head.rotateAngle == 270:
                snek.set_direction('forward')
        if key == 'd':
            if snek.snake_head.rotateAngle == 0:
                snek.set_direction('turn_right')
            if snek.snake_head.rotateAngle == 90:
                snek.set_direction('forward')
            if snek.snake_head.rotateAngle == 180:
                snek.set_direction('turn_left')

    # Slows down the game
    if key == 'left':
        if app.stepsPerSecond == 1:
            print(f'Cannot lower speed past {app.stepsPerSecond / 10}x')
        else:
            app.stepsPerSecond -= 1
            print(f'Lowered speed {app.stepsPerSecond / 10}x')

    # Speeds up the game
    if key == 'right':
        app.stepsPerSecond += 1
        print(f'Increased speed {app.stepsPerSecond / 10}x')

    # Pauses the game
    if key == 'space':
        if isPaused:
            isPaused = False
        else:
            isPaused = True
            print('Paused the game')

    if key == 'G':
        if gridBackground.visible:
            print('Grid hidden')
            gridBackground.visible = False
        else:
            print('Grid shown')
            gridBackground.visible = True

    # Prints information
    if key == 'I':
        # Prints the current grid
        print('Current grid')
        for g in grid:
            print(g)
        print('\n')
        # Prints out the apple seed
        print('appleSeed =', apple.get_seed())

    # Ends the game
    if key == 'E':
        print('Terminated game early')
        gameOver()


def onStep():
    global path
    global isPlaying
    global isPaused
    global grid
    global reset
    global step
    global reward
    global agent
    global best_score
    global total_score
    global plot_scores
    global plot_avg_scores

    if reset:
        resetGame()
        reset = False
        return

    if isPaused:
        return

    if step > 100 * (len(snek.snake_body) + 1):
        gameOver()
        return

    step += 1
    reward = 0

    # Stops path updating the grid and pathfinding if the player is in control
    if not isPlaying:
        grid = genGrid()

        old_state = agent.get_state(grid)

        # get move
        action = agent.get_action(old_state)
        #action = [1, 0, 0]

        if np.array_equal(action, [1, 0, 0]):
            snek.set_direction('forward')
        if np.array_equal(action, [0, 1, 0]):
            snek.set_direction('turn_right')
        if np.array_equal(action, [0, 0, 1]):
            snek.set_direction('turn_left')

    # Moves the snake
    snek.move()

    if snek.snake_head.hits(apple.apple.centerX, apple.apple.centerY):
        reward = 10

        # Adds another body segment
        snek.add_body()

        if score.value == pow(size, 2) - 1:
            gameOver()
            return

        # Determines where the next apple will spawn
        if appleSeed:
            apple.set_apple(appleSeed[0])
            appleSeed.pop(0)
        else:
            apple.gen_apple(snek.snake_head, snek.snake_body)

        apple.update_seed((apple.apple.left, apple.apple.top))

        score.value += 1

    if not isPlaying:
        new_state = agent.get_state(grid)

        agent.train_sm(old_state, action, reward, new_state, snek.is_dead())

        # remember
        agent.remember(old_state, action, reward, new_state, snek.is_dead())

    if snek.is_dead():
        if score.value > best_score:
            best_score = score.value
            agent.model.save()

        curr_score = score.value

        # resets on autoReset
        gameOver()

        # train the long memory
        agent.n_games += 1
        game.value += 1
        agent.train_lm()

        print('Game', agent.n_games, 'Score', curr_score, 'Record', best_score)

        plot_scores.append(curr_score)
        total_score += curr_score
        avg_score = total_score / agent.n_games
        plot_avg_scores.append(avg_score)
        plot(plot_scores, plot_avg_scores)

        return


if __name__ == '__main__':
    cmu_graphics.run()
