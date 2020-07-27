from scipy.interpolate import interp1d
import numpy as np
import pyglet
from parameters import layout_parameters, color_scheme
from board import Board
from stochastic_board import StochasticBoard

class ProbabilityDisplay:
  def __init__(self, board_size):
    self.board_size = board_size
    self.window = pyglet.window.Window(layout_parameters['window_size'], layout_parameters['window_size'])
    self.background = BoardBackground()
    self.painter = BoardPainter(board_size)

  def update(self, probabilities):
    self.window.dispatch_events()
    self.background.draw()
    for x in range(self.board_size):
      for y in range(self.board_size):
        self.draw_tile(x, y, probabilities[x, y])
    self.window.flip()

  def draw_tile(self, x, y, probabilities):
    opacity = self.calculate_opacity(np.max(probabilities))
    dominant_color = np.argmax(probabilities)
    if dominant_color == Board.EMPTY:
      self.painter.draw_intersection(x, y, opacity)
    elif dominant_color == Board.BLACK:
      self.painter.draw_stone(x, y, opacity, color_scheme['black_rgb'])
    elif dominant_color == Board.WHITE:
      self.painter.draw_stone(x, y, opacity, color_scheme['white_rgb'])
    else:
      assert False

  def calculate_opacity(self, highest_probability):
    highest_probability_range = [1/len(StochasticBoard.colors), 1]
    opacity_range = [0, color_scheme['max_opacity']]
    interpolate = interp1d(highest_probability_range, opacity_range)
    return interpolate(highest_probability)

class BoardBackground:
  def __init__(self):
    margin = 10
    background_size = layout_parameters['window_size'] + 2*margin
    self.background = pyglet.shapes.Rectangle(-margin, -margin, width = background_size, height = background_size)
    self.background.color = color_scheme['background_rgb']

  def draw(self):
    self.background.draw()

class BoardPainter:
  def __init__(self, board_size):
    window_size = layout_parameters['window_size']
    self.tile_radius = window_size / (2*board_size)
    self.board_to_screen_x = interp1d([0, board_size-1], [self.tile_radius, window_size - self.tile_radius])
    # y increases upwards in pyglet's coordinates, but downwards in board coordinates
    self.board_to_screen_y = interp1d([0, board_size-1], [window_size - self.tile_radius, self.tile_radius])

  def draw_stone(self, x, y, opacity, color):
    circle = pyglet.shapes.Circle(self.board_to_screen_x(x), self.board_to_screen_y(y), self.tile_radius)
    circle.opacity = opacity
    circle.color = color
    circle.draw() # batching is recommended instead

  def draw_intersection(self, x, y, opacity):
    center_x = self.board_to_screen_x(x)
    center_y = self.board_to_screen_y(y)
    self.draw_vertical_intersection_part(center_x, center_y, opacity)
    self.draw_left_intersection_arm(center_x, center_y, opacity)
    self.draw_right_intersection_arm(center_x, center_y, opacity)

  def draw_vertical_intersection_part(self, center_x, center_y, opacity):
    top_y = center_y + self.tile_radius
    bottom_y = center_y - self.tile_radius
    self.draw_intersection_line(center_x, top_y, center_x, bottom_y, opacity)

  def draw_left_intersection_arm(self, center_x, center_y, opacity):
    left_x = center_x - self.tile_radius
    mid_left_x = center_x - layout_parameters['grid_thickness'] / 2
    self.draw_intersection_line(left_x, center_y, mid_left_x, center_y, opacity)

  def draw_right_intersection_arm(self, center_x, center_y, opacity):
    mid_right_x = center_x + layout_parameters['grid_thickness'] / 2
    right_x = center_x + self.tile_radius
    self.draw_intersection_line(mid_right_x, center_y, right_x, center_y, opacity)

  def draw_intersection_line(self, start_x, start_y, end_x, end_y, opacity):
    line = pyglet.shapes.Line(start_x, start_y, end_x, end_y, width = layout_parameters['grid_thickness'])
    line.color = color_scheme['line_rgb']
    line.opacity = opacity
    line.draw()