import math
import numpy as np
from board import Board

class InputBuilder:
  def __init__(self, model):
    assert model.version == 8
    self.model = model
    self.channel_size = model.pos_len

  # board history and the encore are ignored
  def build_channels(self, board, own_color, rules):
    opponent_color = Board.get_opp(own_color)
    channels = np.zeros(shape = [self.channel_size * self.channel_size, 22], dtype = np.float32)
    self.build_whole_board_channel(channels[:, 0], board)
    self.build_stone_channel(channels[:, 1], board, own_color)
    self.build_stone_channel(channels[:, 2], board, opponent_color)
    self.build_liberty_channel(channels[:, 3], board, 1)
    self.build_liberty_channel(channels[:, 4], board, 2)
    self.build_liberty_channel(channels[:, 5], board, 3)
    assert rules['scoringRule'] == 'SCORING_AREA' # for channels 18, 19
    assert rules['taxRule'] == 'TAX_NONE' # for channels 18, 19
    return prepend_dimension(channels)

  def build_globals(self, board, own_color, rules):
    own_komi = (rules['whiteKomi'] if own_color == Board.WHITE else -rules['whiteKomi'])
    globals = np.zeros(shape = [19], dtype = np.float32)
    globals[5] = own_komi / 20.0
    assert rules['koRule'] == 'KO_SIMPLE' # for globals 6, 7
    globals[8] = rules['multiStoneSuicideLegal']
    assert rules['scoringRule'] == 'SCORING_AREA' # for global 9
    assert rules['taxRule'] == 'TAX_NONE' # for globals 10, 11
    globals[18] = InputBuilder.komi_triangle_wave(own_komi, board)
    return prepend_dimension(globals)

  def build_whole_board_channel(self, channel, board):
    self.build_channel_from_function(channel, board, lambda _: True)

  def build_stone_channel(self, channel, board, color):
    self.build_channel_from_function(channel, board, lambda location: board.board[location] == color)

  def build_liberty_channel(self, channel, board, number_of_liberties):
    self.build_channel_from_function(channel, board, lambda location: board.num_liberties(location) == number_of_liberties)

  # f can return a number, or a bool to be converted
  def build_channel_from_function(self, channel, board, f):
    assert board.size <= self.channel_size
    for x in range(board.size):
      for y in range(board.size):
        position = self.model.xy_to_tensor_pos(x, y)
        location = board.loc(x, y)
        channel[position] = f(location)

  @staticmethod
  def komi_triangle_wave(own_komi, board):
    delta = (own_komi - board.size) % 2
    if delta < 0.5:
      return delta
    elif delta < 1.5:
      return 1.0 - delta
    else:
      return delta - 2.0

def prepend_dimension(array):
  return np.expand_dims(array, 0)