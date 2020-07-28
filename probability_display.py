from scipy.interpolate import interp1d
import numpy as np
import pyglet
from board import Board
from board_colors import board_colors
from stochastic_board import StochasticBoard
from board_graphics import BoardBackground, BoardForeground

class ProbabilityDisplay:
  def __init__(self, window_size, board_size):
    self.window_size = window_size
    self.board_size = board_size
    self.window = pyglet.window.Window(window_size, window_size)
    highest_probability_range = [1/len(board_colors), 1]
    self.highest_probability_to_opacity = interp1d(highest_probability_range, BoardForeground.opacity_range)

  def update(self, probabilities):
    self.window.dispatch_events()
    background = BoardBackground(self.window_size)
    background.draw()
    self.draw_foreground(probabilities)
    self.window.flip()

  def draw_foreground(self, probabilities):
    foreground = BoardForeground(self.window_size, self.board_size)
    for x in range(self.board_size):
      for y in range(self.board_size):
        dominant_color = np.argmax(probabilities[x, y])
        opacity = self.highest_probability_to_opacity(np.max(probabilities[x, y]))
        foreground.add_color(dominant_color, x, y, opacity)
    foreground.draw()