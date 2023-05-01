import matplotlib.pyplot as plt
from IPython import display
import numpy as np


def plot(scores, mean_scores):
    plt.ion()

    display.clear_output(wait=True)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores)
    plt.plot(mean_scores)

    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.pause(0.001)


def display_game(state):
    plt.ion()

    np_grid = np.array(state)

    display.clear_output(wait=True)
    plt.clf()

    plt.axis(False)
    plt.ioff()
    plt.imshow(np_grid)

    plt.pause(0.001)
