import numpy as np
import datatypes
from board import Board

class FullInputBuilder:
  def __init__(self, model):
    assert model.version == 8
    self.model = model
    self.channel_size = model.pos_len

  def build_channels(self, board, own_color, rules):
    opponent_color = Board.get_opp(own_color)
    channels = np.zeros(shape = [self.channel_size * self.channel_size, 22], dtype = datatypes.numpy_float)
    self.build_whole_board_channel(channels[:, 0], board)
    self.build_stone_channel(channels[:, 1], board, own_color)
    self.build_stone_channel(channels[:, 2], board, opponent_color)
    self.build_liberty_channel(channels[:, 3], board, 1)
    self.build_liberty_channel(channels[:, 4], board, 2)
    self.build_liberty_channel(channels[:, 5], board, 3)
    # channel 6 is skipped - ko is ignored
    # channels 7, 8 are skipped - the encore is ignored
    # channels 9-13 are skipped - board history is ignored
    self.build_ladder_channels(channels[:, 14], channels[:, 17], board, own_color)
    # past ladder statuses are set to the current ones because board history is ignored
    channels[:, 15] = channels[:, 14]
    channels[:, 16] = channels[:, 14]
    self.build_pass_alive_area_channels(channels[:, 18], channels[:, 19], board, own_color, rules)
    # channels 20, 21 are skipped - the encore is ignored
    return prepend_dimension(channels)

  def build_whole_board_channel(self, channel, board):
    self.build_channel_from_function(channel, board, lambda _: True)

  def build_stone_channel(self, channel, board, color):
    self.build_channel_from_function(channel, board, lambda location: board.board[location] == color)

  def build_liberty_channel(self, channel, board, number_of_liberties):
    self.build_channel_from_function(channel, board, lambda location: board.num_liberties(location) == number_of_liberties)

  def build_ladder_channels(self, ladderable_stones, working_ladder_captures, board, own_color):
    opponent_color = Board.get_opp(own_color)
    def add_ladderable_stone(location, position, working_moves):
      ladderable_stones[position] = 1.0
      if board.board[location] == opponent_color and 1 < board.num_liberties(location):
        for move in working_moves:
          move_position = self.model.loc_to_tensor_pos(move, board)
          working_ladder_captures[move_position] = 1.0
    self.model.iterLadders(board, add_ladderable_stone)

  def build_pass_alive_area_channels(self, own_area, opponent_area, board, own_color, rules):
    assert rules['scoringRule'] == 'SCORING_AREA'
    assert rules['taxRule'] == 'TAX_NONE'
    opponent_color = Board.get_opp(own_color)
    area = [0 for _ in range(board.arrsize)]
    board.calculateArea(area, True, True, True, rules['multiStoneSuicideLegal'])
    for x in range(board.size):
      for y in range(board.size):
        position = self.model.xy_to_tensor_pos(x, y)
        if area[board.loc(x, y)] == own_color:
          own_area[position] = 1.0
        elif area[board.loc(x, y)] == opponent_color:
          opponent_area[position] = 1.0

  # f can return a number, or a bool to be converted
  def build_channel_from_function(self, channel, board, f):
    assert board.size <= self.channel_size
    for x in range(board.size):
      for y in range(board.size):
        position = self.model.xy_to_tensor_pos(x, y)
        location = board.loc(x, y)
        channel[position] = f(location)

  def build_globals(self, board, own_color, rules):
    own_komi = (rules['whiteKomi'] if own_color == Board.WHITE else -rules['whiteKomi'])
    globals = np.zeros(shape = [19], dtype = datatypes.numpy_float)
    # globals 0-4 are skipped - board history is ignored
    globals[5] = own_komi / 20.0
    assert rules['koRule'] == 'KO_SIMPLE' # for globals 6, 7
    globals[8] = rules['multiStoneSuicideLegal']
    assert rules['scoringRule'] == 'SCORING_AREA' # for global 9
    assert rules['taxRule'] == 'TAX_NONE' # for globals 10, 11
    # globals 12-14 are skipped - the encore and game phases are ignored
    # globals 15, 16 are skipped - playout doubling advantage is ignored
    # global 17 is skipped - button go is ignored
    globals[18] = FullInputBuilder.komi_triangle_wave(own_komi, board.size)
    return prepend_dimension(globals)

  @staticmethod
  def komi_triangle_wave(own_komi, board_size):
    delta = (own_komi - board_size) % 2
    if delta < 0.5:
      return delta
    elif delta < 1.5:
      return 1.0 - delta
    else:
      return delta - 2.0

class QuickInputBuilder(FullInputBuilder):
  def build_ladder_channels(self, *_):
    pass

  def build_pass_alive_area_channels(self, *_):
    pass

def prepend_dimension(array):
  return np.expand_dims(array, 0)
