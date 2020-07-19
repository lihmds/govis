import random
import itertools
import scipy.stats
import numpy as np
from board import Board

'''A square board with a probability distribution instead of a fixed color at each intersection.
All arrays are indexed like array[x, y] (column-major).'''

class StochasticBoard:
  '''The possible colors of an intersection.
  They can be used for array indexing because they are simply 0, 1 and 2.'''
  colors = [Board.EMPTY, Board.BLACK, Board.WHITE]
  assert(colors == [0, 1, 2])

  def __init__(self, size):
    '''Create a board with uniform distributions everywhere.'''
    self.size = size
    # float32 is what model.py uses
    self.logits = np.zeros([size, size, len(StochasticBoard.colors)], dtype = np.float32)

  def probabilities(self):
    '''Return the distributions of all intersections as a size × size × 3 array.
    Indexed as array[x, y, color].'''
    relative_probabilities = np.exp(self.logits)
    return relative_probabilities / relative_probabilities.sum(axis = 2, keepdims = True)

  def entropies(self):
    '''Return the entropy of each probability distribution in nats as a size × size array.'''
    return scipy.stats.entropy(np.exp(self.logits), axis = 2)

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
    return random.choices(population = StochasticBoard.colors, weights = relative_probabilities)[0]

  # table[x][y][color] is a list of evaluations of boards with color at (x, y)
  # might become a separate class
  def create_evaluation_table(self):
    return np.zeros(shape = [self.size, self.size, len(StochasticBoard.colors), 0]).tolist()

  def add_board_to_evaluation_table(self, board, table, objective_function):
    evaluation = objective_function(board)
    for x in range(self.size):
      for y in range(self.size):
        color = board.board[board.loc(x, y)]
        table[x][y][color].append(evaluation)

  def average_evaluation_table(self, table, average_of_empty):
    averages = np.zeros_like(self.logits)
    for x in range(self.size):
      for y in range(self.size):
        for color in StochasticBoard.colors:
          averages[x][y][color] = np.mean(table[x][y][color])
    return np.nan_to_num(averages, copy = False, nan = average_of_empty)

  def estimate_gradient(self, objective_function, sample):
    table = self.create_evaluation_table()
    for board in sample:
      self.add_board_to_evaluation_table(board, table, objective_function)
    overall_evaluation = np.mean(list(itertools.chain.from_iterable(table[0][0])))
    average_evaluations = self.average_evaluation_table(table, average_of_empty = overall_evaluation)
    return self.probabilities() * (average_evaluations - overall_evaluation)