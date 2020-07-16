import math
import tensorflow as tf
import numpy as np
from board import Board

class InputBuilder:
  # f can return a number, or a bool to be converted
  def build_channel_from_function(self, channel, channel_size, board, f):
    for y in range(board.size):
      for x in range(board.size):
        pos = xy_to_tensor_pos(x, y, channel_size)
        location = board.loc(x, y)
        channel[pos] = f(location)

  def build_whole_board_channel(self, channel, channel_size, board):
    self.build_channel_from_function(channel, channel_size, board, lambda _: True)

  def build_channel_of_color(self, channel, channel_size, board, color):
    self.build_channel_from_function(channel, channel_size, board, lambda location: board.board[location] == color)

  def build_channel_of_liberties(self, channel, channel_size, board, number_of_liberties):
    self.build_channel_from_function(channel, channel_size, board, lambda location:
                                     board.num_liberties(location) == number_of_liberties)

  def build(self, model, board, own_color, channel_size, rules):
    assert(model.version == 8)
    assert(board.size <= channel_size)
    assert(rules['encorePhase'] == 0)
    assert(rules['scoringRule'] == 'SCORING_AREA')
    assert(rules['koRule'] == 'KO_SIMPLE')
    assert(rules['taxRule'] == 'TAX_NONE')
    assert(rules['passWouldEndPhase'] == False)

    num_channel_input_features = 22
    channel_input = np.zeros(shape = [channel_size * channel_size, num_channel_input_features], dtype = np.float32)

    num_global_input_features = 19
    global_input = np.zeros(shape = [num_global_input_features], dtype = np.float32)

    opponent_color = Board.get_opp(own_color)
    white_komi = rules['whiteKomi']
    own_komi = (white_komi if own_color == Board.WHITE else -white_komi)

    self.build_whole_board_channel(channel_input[:,0], channel_size, board)
    self.build_channel_of_color(channel_input[:,1], channel_size, board, own_color)
    self.build_channel_of_color(channel_input[:,2], channel_size, board, opponent_color)
    self.build_channel_of_liberties(channel_input[:,3], channel_size, board, 1)
    self.build_channel_of_liberties(channel_input[:,4], channel_size, board, 2)
    self.build_channel_of_liberties(channel_input[:,5], channel_size, board, 3)
    global_input[5] = own_komi / 20.0
    global_input[8] = rules['multiStoneSuicideLegal']
    global_input[18] = self.komi_sawtooth_wave(board, own_komi)
    return prepend_dimension(channel_input), prepend_dimension(global_input)

  def komi_sawtooth_wave(self, board, self_komi):
    board_area_is_even = board.size % 2 == 0
    drawable_komis_are_even = board_area_is_even

    if drawable_komis_are_even:
      komi_floor = math.floor(self_komi / 2.0) * 2.0
    else:
      komi_floor = math.floor((self_komi-1.0) / 2.0) * 2.0 + 1.0

    delta = self_komi - komi_floor
    assert(-0.0001 <= delta)
    assert(delta <= 2.0001)
    delta = clamp(0.0, delta, 2.0)

    if delta < 0.5:
      return delta
    elif delta < 1.5:
      return 1.0 - delta
    else:
      return delta - 2.0

def prepend_dimension(array):
  return np.expand_dims(array, 0)

def clamp(min, x, max):
  if x < min:
    return min
  elif max < x:
    return max
  else:
    return x

def xy_to_tensor_pos(x, y, channel_size):
  return y*channel_size + x

def has_stone(intersection):
  return intersection == Board.BLACK or intersection == Board.WHITE