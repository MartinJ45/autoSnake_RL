# Name: Martin Jimenez
# Date: 03/17/2023 (last updated)

from cmu_graphics import *
from snake_classes import Snake
from snake_classes import Apple
import numpy as np

# The default game speed is 10; this value can be changed by pressing the
# left and right arrow keys
app.stepsPerSecond = 10
step = 0

isPaused = False
isPlaying = False
reset = False
autoReset = False

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

path = []

appleSeed = []
snek = Snake(blockSize, blockSize, blockSize, size)
apple = Apple(200, blockSize, blockSize, size)

reward = 0

gameOverMessage = Label('GAME OVER', 200, 200, size=50, fill='red', visible=False)

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

    snek = Snake(blockSize, blockSize, blockSize, size)
    apple = Apple(200, blockSize, blockSize, size)

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

    if reset:
        resetGame()
        reset = False
        return

    if isPaused:
        return

    if snek.is_dead():
        gameOver()
        return

    if step > 100 * (len(snek.snake_body) + 1):
        gameOver()
        return

    step += 1
    reward = 0

    # Stops path updating the grid and pathfinding if the player is in control
    if not isPlaying:
        # Updates the grid whenever the snake doesn't have a path to follow
        # The game would run significantly slower if the grid were to be
        # updated every single step
        grid = genGrid()

        action = [1, 0, 0]

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


if __name__ == '__main__':
    cmu_graphics.run()