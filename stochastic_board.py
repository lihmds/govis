import random
import itertools
import numpy as np
from datatypes import NumpyFloat
from board import Board
from board_colors import board_colors

class StochasticBoard:
  '''A square board with a probability distribution instead of a fixed color at each intersection.'''

  def __init__(self, size):
    '''Create a board with uniform distributions everywhere.'''
    self.size = size
    self.logits = np.zeros(shape = [size, size, len(board_colors)], dtype = NumpyFloat)

  def probabilities(self):
    '''Return the distributions of all intersections as an array with shape [size, size, 3].
    Indexed as array[x, y, color].'''
    relative_probabilities = np.exp(self.logits)
    return relative_probabilities / relative_probabilities.sum(axis = 2, keepdims = True)

  def ascend_gradient(self, objective_function, rate, sample_size):
    '''Adjust the probabilities so that the expected value of objective_function grows.'''
    sample = [self.generate_board() for _ in range(sample_size)]
    gradient = self.estimate_gradient(objective_function, sample)
    self.logits += rate * gradient

  def generate_board(self):
    '''Generate a board which approximately follows self.probabilities().
    Captures and suicides are not prevented during the generation.
    Because of that, the actual distribution can be different than expected.'''
    # the board is generated column by column, so only the last move can be a suicide
    board = Board(self.size)
    for x in range(self.size):
      for y in range(self.size):
        color = self.generate_color(x, y)
        if board.would_be_legal(color, board.loc(x, y)):
          board.play(color, board.loc(x, y))
    return board

  def generate_color(self, x, y):
    relative_probabilities = np.exp(self.logits[x, y])
    return random.choices(population = board_colors, weights = relative_probabilities)[0]

  def estimate_gradient(self, objective_function, sample):
    evaluations = EvaluationTable(self.size)
    for board in sample:
      evaluations.add_board(board, objective_function(board))
    overall_evaluation = evaluations.global_average()
    local_evaluations = evaluations.local_averages(average_of_empty = overall_evaluation)
    return self.probabilities() * (local_evaluations - overall_evaluation)

class EvaluationTable:
  def __init__(self, size):
    self.size = size
    # self.table[x][y][color] is a list of evaluations of boards with color at (x, y)
    # lists are used because the last dimension is rugged
    self.table = np.zeros(shape = [size, size, len(board_colors), 0]).tolist()

  def add_board(self, board, evaluation):
    for x in range(self.size):
      for y in range(self.size):
        color = board.board[board.loc(x, y)]
        self.table[x][y][color].append(evaluation)

  def global_average(self):
    return np.mean(list(itertools.chain.from_iterable(self.table[0][0])))

  def local_averages(self, average_of_empty):
    old_settings = np.seterr(invalid = 'ignore')
    averages = np.zeros(shape = [self.size, self.size, len(board_colors)], dtype = NumpyFloat)
    for x in range(self.size):
      for y in range(self.size):
        for color in board_colors:
          averages[x, y, color] = np.mean(self.table[x][y][color])
    np.seterr(**old_settings)
    return np.nan_to_num(averages, copy = False, nan = average_of_empty)
