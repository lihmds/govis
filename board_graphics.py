from scipy.interpolate import interp1d
import pyglet
from board import Board

color_scheme = {
  'wood': (250, 170, 80),
  'black': (0, 0, 0),
  'white': (255, 255, 255),
  'line': (0, 0, 0)
}

class BoardBackground:
  def __init__(self, window_size):
    margin = 10
    background_size = window_size + 2*margin
    self.background = pyglet.shapes.Rectangle(-margin, -margin, width = background_size, height = background_size)
    self.background.color = color_scheme['wood']

  def draw(self):
    self.background.draw()

class BoardForeground:
  opacity_range = [0, 255]

  def __init__(self, window_size, board_size):
    self.tile_radius = window_size / (2*board_size)
    self.board_to_screen_x = interp1d([0, board_size-1], [self.tile_radius, window_size - self.tile_radius])
    # y increases upwards in pyglet's coordinates, but downwards in board coordinates
    self.board_to_screen_y = interp1d([0, board_size-1], [window_size - self.tile_radius, self.tile_radius])
    self.grid_thickness = 2
    self.batch = pyglet.graphics.Batch()
    # A shape will only draw if we keep a reference to it. I can't explain it.
    # Other than that, we don't need a list of the shapes.
    self.batched_shapes = []

  def draw(self):
    self.batch.draw()

  def add_color(self, board_color, x, y, opacity):
    if board_color == Board.EMPTY:
      self.add_intersection(x, y, opacity)
    elif board_color == Board.BLACK:
      self.add_stone(x, y, opacity, color_scheme['black'])
    elif board_color == Board.WHITE:
      self.add_stone(x, y, opacity, color_scheme['white'])
    else:
      assert False

  def add_intersection(self, x, y, opacity):
    center_x = self.board_to_screen_x(x)
    center_y = self.board_to_screen_y(y)
    self.add_left_intersection_arm(center_x, center_y, opacity)
    self.add_vertical_intersection_part(center_x, center_y, opacity)
    self.add_right_intersection_arm(center_x, center_y, opacity)

  def add_left_intersection_arm(self, center_x, center_y, opacity):
    left_x = center_x - self.tile_radius
    mid_left_x = center_x - self.grid_thickness/2
    self.add_intersection_line(left_x, center_y, mid_left_x, center_y, opacity)

  def add_vertical_intersection_part(self, center_x, center_y, opacity):
    top_y = center_y + self.tile_radius
    bottom_y = center_y - self.tile_radius
    self.add_intersection_line(center_x, top_y, center_x, bottom_y, opacity)

  def add_right_intersection_arm(self, center_x, center_y, opacity):
    mid_right_x = center_x + self.grid_thickness/2
    right_x = center_x + self.tile_radius
    self.add_intersection_line(mid_right_x, center_y, right_x, center_y, opacity)

  def add_intersection_line(self, start_x, start_y, end_x, end_y, opacity):
    line = pyglet.shapes.Line(start_x, start_y, end_x, end_y,
                              width = self.grid_thickness, batch = self.batch)
    line.opacity = opacity
    line.color = color_scheme['line']
    self.batched_shapes.append(line)

  def add_stone(self, x, y, opacity, color):
    stone = pyglet.shapes.Circle(self.board_to_screen_x(x), self.board_to_screen_y(y),
                                 radius = self.tile_radius, batch = self.batch)
    stone.opacity = opacity
    stone.color = color
    self.batched_shapes.append(stone)