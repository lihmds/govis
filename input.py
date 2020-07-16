import math
import tensorflow as tf
import numpy as np
from board import Board

def generate_input(model, board, own_color, channel_size, rules):
  assert(model.version == 8)
  assert(board.size <= channel_size)
  assert(rules['encorePhase'] == 0)
  assert(rules['scoringRule'] == 'SCORING_AREA')
  assert(rules['koRule'] == 'KO_SIMPLE')
  assert(rules['taxRule'] == 'TAX_NONE')
  assert(rules['passWouldEndPhase'] == False)

  num_channel_input_features = 22
  channel_input_shape = [channel_size * channel_size, num_channel_input_features]
  channel_input = np.zeros(shape = channel_input_shape, dtype = np.float32)

  num_global_input_features = 19
  global_input_shape = [num_global_input_features]
  global_input = np.zeros(shape = global_input_shape, dtype = np.float32)

  opponent_color = Board.get_opp(own_color)

  for y in range(board.size):
    for x in range(board.size):
      pos = xy_to_tensor_pos(x, y, channel_size)
      channel_input[pos,0] = 1.0 # location is on the board
      intersection = board.board[board.loc(x, y)]
      if intersection == own_color:
        channel_input[pos,1] = 1.0 # location has own stone
      elif intersection == opponent_color:
        channel_input[pos,2] = 1.0 # location has opponent stone

      if has_stone(intersection):
        libs = board.num_liberties(loc)
        if libs == 1:
          channel_input[pos,3] = 1.0 # has chain with 1 liberty
        elif libs == 2:
          channel_input[pos,4] = 1.0 # has chain with 2 liberties
        elif libs == 3:
          channel_input[pos,5] = 1.0 # has chain with 3 liberties

  # channel 6 is set to 1 where a ko forbids the move
  # channels 7 and 8 are related to the encore
  # channels 9-13 are set to 1 where the last five moves were played (if such a place exists)
  # channel 9 is the most recent one

  # assumed: channels 14-16 describe ladderable stones 0,1,2 turns ago
  # (i don't know if a turn means one or two moves)
  # assumed: channel 17 are moves which start a successful ladder
  # channels 18 and 19 describe pass-alive alive for both players
  # channels 20 and 21 are second encore phase starting stones

  board_area = board.size * board.size
  white_komi = rules['whiteKomi']
  self_komi = (white_komi if own_color == Board.WHITE else -white_komi)

  # globals 0-4 describe the last five moves
  # global 0 is the most recent
  # they are set to 1 if the move was a pass

  # global 5 is the komi (from katago's perspective) scaled down by 20
  # (no longer by 15 as 'Accelerating Self-Play Learning in Go' describes)
  global_input[5] = self_komi / 20.0

  # globals 6 and 7 encode the ko ruleset
  # global 8 encodes the suicide rule
  if rules['multiStoneSuicideLegal']:
    global_input[8] = 1.0

  # global 9 is set to 1 if territory scoring is used
  # globals 10 and 11 encode the tax rule
  # globals 12 and 13 describe the encore
  # global 14 is set to 1 if a pass would end the phase
  # asssumed: globals 15 and 16 describe the playout doubling advantage
  # global 17 is 1 if the 'button' from button go is available
  # global 18 is parity information about the komi relative to the board
  global_input[18] = komi_sawtooth_wave(board, self_komi)
  return prepend_dimension(channel_input), prepend_dimension(global_input)

def komi_sawtooth_wave(board, self_komi):
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