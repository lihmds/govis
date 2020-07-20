import math
import numpy as np
from board import Board

class InputBuilder:
  # board history and the encore are ignored
  def build_channels(self, model, board, own_color, rules):
    assert model.version == 8
    channel_size = model.pos_len
    opponent_color = Board.get_opp(own_color)
    channel_input = np.zeros(shape = [channel_size * channel_size, 22], dtype = np.float32)
    self.build_whole_board_channel(channel_input[:, 0], channel_size, board)
    self.build_stone_channel(channel_input[:, 1], channel_size, board, own_color)
    self.build_stone_channel(channel_input[:, 2], channel_size, board, opponent_color)
    self.build_liberty_channel(channel_input[:, 3], channel_size, board, 1)
    self.build_liberty_channel(channel_input[:, 4], channel_size, board, 2)
    self.build_liberty_channel(channel_input[:, 5], channel_size, board, 3)
    assert rules['scoringRule'] == 'SCORING_AREA' # for channels 18, 19
    assert rules['taxRule'] == 'TAX_NONE' # for channels 18, 19
    return prepend_dimension(channel_input)

  def build_globals(self, model, board, own_color, rules):
    assert model.version == 8
    own_komi = (rules['whiteKomi'] if own_color == Board.WHITE else -rules['whiteKomi'])
    global_input = np.zeros(shape = [19], dtype = np.float32)
    global_input[5] = own_komi / 20.0
    assert rules['koRule'] == 'KO_SIMPLE' # for globals 6, 7
    global_input[8] = rules['multiStoneSuicideLegal']
    assert rules['scoringRule'] == 'SCORING_AREA' # for global 9
    assert rules['taxRule'] == 'TAX_NONE' # for globals 10, 11
    global_input[18] = self.komi_triangle_wave(board, own_komi)
    return prepend_dimension(global_input)

  def build_whole_board_channel(self, channel, channel_size, board):
    self.build_channel_from_function(channel, channel_size, board, lambda _: True)

  def build_stone_channel(self, channel, channel_size, board, color):
    self.build_channel_from_function(channel, channel_size, board, lambda location: board.board[location] == color)

  def build_liberty_channel(self, channel, channel_size, board, number_of_liberties):
    self.build_channel_from_function(channel, channel_size, board, lambda location:
                                     board.num_liberties(location) == number_of_liberties)

  # f can return a number, or a bool to be converted
  def build_channel_from_function(self, channel, channel_size, board, f):
    assert board.size <= channel_size
    for x in range(board.size):
      for y in range(board.size):
        position = xy_to_tensor_position(x, y, channel_size)
        location = board.loc(x, y)
        channel[position] = f(location)

  def komi_triangle_wave(self, board, own_komi):
    if is_even(board.size):
      komi_floor = math.floor(own_komi / 2.0) * 2.0
    else:
      komi_floor = math.floor((own_komi-1.0) / 2.0) * 2.0 + 1.0

    delta = own_komi - komi_floor
    assert -0.0001 <= delta
    assert delta <= 2.0001
    delta = clamp(0.0, delta, 2.0)

    if delta < 0.5:
      return delta
    elif delta < 1.5:
      return 1.0 - delta
    else:
      return delta - 2.0

def prepend_dimension(array):
  return np.expand_dims(array, 0)

def xy_to_tensor_position(x, y, channel_size):
  return y*channel_size + x

def is_even(n):
  return n % 2 == 0

def clamp(min, x, max):
  if x < min:
    return min
  elif max < x:
    return max
  else:
    return x