from scipy.interpolate import interp1d
import numpy as np
import pyglet
from board import Board
from stochastic_board import StochasticBoard

class BoardProbabilityDisplay:
  def __init__(self, board_size, window_size):
    self.board_size = board_size
    self.window_size = window_size
    self.tile_radius = window_size / (2 * board_size)
    self.window = pyglet.window.Window(window_size, window_size)
    self.background_color = (250, 170, 80)
    self.black_color = (0, 0, 0)
    self.white_color = (255, 255, 255)

  def draw(self, probabilities):
    self.window.dispatch_events()
    self.draw_background()
    for x in range(self.board_size):
      for y in range(self.board_size):
        self.draw_tile(x, y, probabilities[x, y])
    self.window.flip()

  def draw_background(self):
    margin = 10
    background_size = self.window_size + 2*margin
    background = pyglet.shapes.Rectangle(-margin, -margin, width = background_size, height = background_size)
    background.color = self.background_color
    background.draw()

  def draw_tile(self, x, y, probabilities):
    center = self.tile_center(x, y)
    opacity = self.calculate_opacity(probabilities)
    dominant_color = np.argmax(probabilities)
    if dominant_color == Board.BLACK:
      self.draw_stone(center, opacity, self.black_color)
    elif dominant_color == Board.WHITE:
      self.draw_stone(center, opacity, self.white_color)
    elif dominant_color == Board.EMPTY:
      self.draw_intersection(center, opacity)
    else:
      assert False

  def draw_stone(self, center, opacity, color):
    window_x, window_y = center
    circle = pyglet.shapes.Circle(window_x, window_y, radius = self.tile_radius)
    circle.opacity = opacity
    circle.color = color
    circle.draw() # batching is recommended instead

  def draw_intersection(self, center, opacity):
    pass

  def tile_center(self, x, y):
    interpolate_x = interp1d([0, self.board_size-1], [self.tile_radius, self.window_size-self.tile_radius])
    # y increases upwards in pyglet's coordinates, but downwards in board coordinates
    interpolate_y = interp1d([0, self.board_size-1], [self.window_size-self.tile_radius, self.tile_radius])
    return interpolate_x(x), interpolate_y(y)

  def calculate_opacity(self, probabilities):
    interpolate = interp1d([1/len(StochasticBoard.colors), 1], [0, 255])
    return interpolate(np.max(probabilities))