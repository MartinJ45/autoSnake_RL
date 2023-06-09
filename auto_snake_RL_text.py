# Name: Martin Jimenez
# Date: 05/01/2023 (last updated)

from random import *
import numpy as np
import shelve
import time

from agent import Agent
from helper import plot, display_game

# The default game speed is 10
stepsPerSecond = 10
step = 0

# Default boolean values
isPaused = False    # not in use
isPlaying = False   # not in use
reset = False       # tells the program to reset the game when it ends
autoReset = True    # automatically resets the game when it ends
autoSave = False    # automatically saves the model every 100 games
add_body = False    # tells the program to add a snake piece when the snake eats an apple

size = 18           # the size of the playable board (18x18) (with the border: 20x20)

# This is the grid that the snake uses
# 0 represents open space
# 1 represents the border or the snake body
# 3 represents the snake head
# 5 is a goal representing the end of the tail (used when it cannot locate the apple)
# 9 is a goal representing the apple
grid = [
    [1] * (size + 2)
]

for i in range(size):
    grid.append([1] + [0] * size + [1])
grid.append([1] * (size + 2))

# snake dictionary {(positionY, positionX): (direction)}
# (0, 1) -> moving right
# (0, -1) -> moving left
# (1, 0) -> moving down
# (-1, 0) -> moving up
snek = {(9, 8): (0, 1)}

# action the snake is taking
# [1, 0, 0] -> straight
# [0, 1, 0] -> turn right
# [0, 0, 1] -> turn left
action = [1, 0, 0]

# apple position
apple = (9, 9)

#
appleSeed = []

reward = 0

# used when plotting
plot_scores = []
plot_avg_scores = []
total_score = 0
score = 0

agent = Agent()
n_games, best_score = agent.model.load()
agent.n_games = int(n_games)
initial_games = int(n_games)
best_score = int(best_score)


# Stops the program when the snake loses, wins, or if the game is ended early
def game_over():
    if autoReset:
        reset_game()
        return


# resets variables for a new game
def reset_game():
    global appleSeed
    global snek
    global apple
    global step
    global score

    appleSeed = []
    snek = {(9, 8): (0, 1)}
    apple = (9, 9)

    score = 0
    step = 0


# Updates the grid
def gen_grid(snek, apple):
    grid = [
        [1] * (size + 2)
    ]

    for i in range(size):
        grid.append([1] + [0] * size + [1])
    grid.append([1] * (size + 2))

    for y, x in list(snek):
        grid[y][x] = 1

    grid[list(snek)[0][0]][list(snek)[0][1]] = 3

    grid[apple[0]][apple[1]] = 9

    return grid


# moves the snake across the board
def move_snake(snek, add_body):
    if add_body:
        last_pos = list(snek)[-1]
        last_change = snek.get(last_pos)

    for i in range(len(snek)):
        pos = list(snek)[0]
        change = snek.get(pos)

        new_pos = pos[0] + change[0], pos[1] + change[1]

        if snek.get(new_pos):
            return snek, False, True

        if i == 0:
            new_change = change
        else:
            new_change = prev_change

        prev_change = change
        snek.update({new_pos: new_change})
        snek.pop(pos)

    if add_body:
        snek.update({last_pos: last_change})

    return snek, False, False


# generates a new apple when one is eaten
def gen_apple(snek):
    global isPaused

    x = randint(1, size)
    y = randint(1, size)

    apple = (y, x)

    for pos in list(snek):
        if pos == apple:
            apple = gen_apple(snek)

    return apple


# saves the model
def save():
    global agent

    agent.model.save(agent.n_games, best_score)
    agent.forget(step)
    agent.memory.close()
    agent.memory = shelve.open('model/memory.pickle', writeback=True)
    print('Saving the model')


'''
def onKeyPress(key):
    global isPaused
    global isPlaying
    global reset
    global autoReset
    global action
    global best_score
    global step

    if key == 'S':
        agent.model.save(agent.n_games, best_score)
        agent.forget(step)
        agent.memory.close()
        agent.memory = shelve.open('model/memory.pickle', writeback=True)
        print('Saving the model')

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
                #snek.set_direction('forward')
                action = [1, 0, 0]
            if snek.snake_head.rotateAngle == 90:
                action = [0, 0, 1]
                #snek.set_direction('turn_left')
            if snek.snake_head.rotateAngle == 270:
                action = [0, 1, 0]
                #snek.set_direction('turn_right')
        if key == 's':
            if snek.snake_head.rotateAngle == 180:
                #snek.set_direction('forward')
                action = [1, 0, 0]
            if snek.snake_head.rotateAngle == 90:
                #snek.set_direction('turn_right')
                action = [0, 1, 0]
            if snek.snake_head.rotateAngle == 270:
                #snek.set_direction('turn_left')
                action = [0, 0, 1]
        if key == 'a':
            if snek.snake_head.rotateAngle == 0:
                #snek.set_direction('turn_left')
                action = [0, 0, 1]
            if snek.snake_head.rotateAngle == 180:
                #snek.set_direction('turn_right')
                action = [0, 1, 0]
            if snek.snake_head.rotateAngle == 270:
                #snek.set_direction('forward')
                action = [1, 0, 0]
        if key == 'd':
            if snek.snake_head.rotateAngle == 0:
                #snek.set_direction('turn_right')
                action = [0, 1, 0]
            if snek.snake_head.rotateAngle == 90:
                #snek.set_direction('forward')
                action = [1, 0, 0]
            if snek.snake_head.rotateAngle == 180:
                #snek.set_direction('turn_left')
                action = [0, 0, 1]

    # Slows down the game
    if key == 'left':
        if app.stepsPerSecond <= 1:
            app.stepsPerSecond /= 2
        else:
            app.stepsPerSecond -= 1
        print(f'Lowered speed {app.stepsPerSecond / 10}x')

    # Speeds up the game
    if key == 'right':
        if app.stepsPerSecond <= 1:
            app.stepsPerSecond *= 2
        else:
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
        grid = genGrid()
        for g in grid:
            print(g)
        print('\n')

        print('Current state')
        print(agent.get_state(grid, snek, apple, border, blockSize))

        # Prints out the apple seed
        print('appleSeed =', apple.get_seed())

    # Ends the game
    if key == 'E':
        print('Terminated game early')
        gameOver()
'''


def play_step(delay=0.0, train=True, plot_score=True, display=False):
    global snek
    global apple
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
    global action
    global autoSave
    global score
    global add_body

    time.sleep(delay)

    if reset:
        reset = False
        return

    if isPaused:
        return

    if step > 50 * len(snek):
        reward = -5
    else:
        reward = 0

    grid = gen_grid(snek, apple)

    # print('grid:')
    # for g in grid:
    #     print(g)
    # print('\n')

    old_state = agent.get_state(grid, snek, apple)

    if not isPlaying:
        action = agent.get_action(old_state)

    change = snek.get(list(snek)[0])

    if np.array_equal(action, [1, 0, 0]):   # straight
        change = change
    if np.array_equal(action, [0, 1, 0]):   # turn right
        if change == (0, 1):        # right -> down
            change = (1, 0)
        elif change == (1, 0):      # down -> left
            change = (0, -1)
        elif change == (0, -1):     # left -> up
            change = (-1, 0)
        elif change == (-1, 0):     # up -> right
            change = (0, 1)
    if np.array_equal(action, [0, 0, 1]):   # turn left
        if change == (0, 1):        # right -> up
            change = (-1, 0)
        elif change == (-1, 0):     # up -> left
            change = (0, -1)
        elif change == (0, -1):     # left -> down
            change = (1, 0)
        elif change == (1, 0):      # down -> right
            change = (0, 1)

    snek.update({list(snek)[0]: change})

    snek, add_body, is_dead = move_snake(snek, add_body)
    # grid = gen_grid(snek, apple)

    if isPlaying:
        action = [1, 0, 0]

    if list(snek)[0] == apple:
        reward = 20             # reward for eating apple

        # Adds another body segment
        add_body = True

        if score == pow(size, 2) - 1:
            # gameOver()
            return

        # Determines where the next apple will spawn
        if appleSeed:
            apple = appleSeed[0]
            appleSeed.pop(0)
        else:
            apple = gen_apple(snek)

        # appleSeed.append(apple)

        score += 1

    if is_dead or list(snek)[0][0] in (0, 19) or list(snek)[0][1] in (0, 19) or step > 100 * len(snek):
        reward = -20

        if autoSave and agent.n_games % 100 == 0:
            save()

        if score > best_score:
            best_score = score
        # if score < best_score - 10:
        #     agent.forget(step)

        curr_score = score

        grid = gen_grid(snek, apple)

        # trains short memory
        new_state = agent.get_state(grid, snek, apple)

        if train:
            agent.train_sm(old_state, action, reward, new_state, game_over=True)

            # remember
            agent.remember(old_state, action, reward, new_state, game_over=True)

            # game.value += 1
            agent.train_lm()
        else:
            agent.test_sm(old_state, action, reward, new_state, game_over=True)

            agent.test_lm()

        # train the long memory
        agent.n_games += 1

        print('Game', agent.n_games, 'Score', curr_score, 'Record', best_score)

        plot_scores.append(curr_score)
        total_score += curr_score
        avg_score = total_score / (agent.n_games - initial_games)
        plot_avg_scores.append(avg_score)
        if plot_score:
            plot(plot_scores, plot_avg_scores)

        # resets on autoReset
        game_over()
    else:
        grid = gen_grid(snek, apple)

        new_state = agent.get_state(grid, snek, apple)

        if train:
            agent.train_sm(old_state, action, reward, new_state, game_over=False)

            # remember
            agent.remember(old_state, action, reward, new_state, game_over=False)
        else:
            agent.test_sm(old_state, action, reward, new_state, game_over=False)

    if display:
        display_game(grid)

    step += 1


if __name__ == '__main__':
    print(len(agent.memory['mem']))

    # print(agent.memory['mem'])

    games = 1000

    num_start = agent.n_games

    timer_start = time.time()

    while agent.n_games - num_start < games:
        play_step(delay=0, train=True, plot_score=False, display=False)

    timer_end = time.time()

    total_time = timer_end - timer_start

    print(f'Took {total_time:.2f}s to train for {games} games')
    print(f'Averaged {total_time / games:.2f}s per game')
    save()
    input()
