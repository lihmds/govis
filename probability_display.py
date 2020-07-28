from scipy.interpolate import interp1d
import numpy as np
import pyglet
from board import Board
from stochastic_board import StochasticBoard
from board_graphics import BoardBackground, BoardPainter

class ProbabilityDisplay:
  def __init__(self, window_size, board_size):
    self.window_size = window_size
    self.board_size = board_size
    self.window = pyglet.window.Window(window_size, window_size)
    self.painter = BoardPainter(window_size, board_size)
    highest_probability_range = [1/len(StochasticBoard.colors), 1]
    self.highest_probability_to_opacity = interp1d(highest_probability_range, BoardPainter.opacity_range)

  def update(self, probabilities):
    self.window.dispatch_events()
    background = BoardBackground(self.window_size)
    background.draw()
    self.draw_tiles(probabilities)
    self.window.flip()

  def draw_tiles(self, probabilities):
    self.painter.begin_batch()
    for x in range(self.board_size):
      for y in range(self.board_size):
        self.draw_tile(x, y, probabilities[x, y])
    self.painter.end_batch()

  def draw_tile(self, x, y, probabilities):
    opacity = self.highest_probability_to_opacity(np.max(probabilities))
    dominant_color = np.argmax(probabilities)
    if dominant_color == Board.EMPTY:
      self.painter.draw_intersection(x, y, opacity)
    elif dominant_color == Board.BLACK:
      self.painter.draw_black_stone(x, y, opacity)
    elif dominant_color == Board.WHITE:
      self.painter.draw_white_stone(x, y, opacity)
    else:
      assert False