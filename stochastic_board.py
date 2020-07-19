import random
import itertools
import scipy.stats
import numpy as np
from board import Board

class StochasticBoard:
  # we can index arrays based on those without any conversion because they are simply 0, 1, 2
  colors = [Board.EMPTY, Board.BLACK, Board.WHITE]
  assert(colors == [0, 1, 2])

  def __init__(self, size):
    self.size = size
    self.logits = np.zeros([size, size, len(StochasticBoard.colors)], dtype = np.float32)

  def probabilities(self):
    relative_probabilities = np.exp(self.logits)
    return relative_probabilities / relative_probabilities.sum(axis = 2, keepdims = True)

  def entropies(self):
    return scipy.stats.entropy(np.exp(self.logits), axis = 2)

  def ascend_gradient(self, objective_function, rate, sample_size):
    boards = [self.generate_board() for _ in range(sample_size)]
    gradient = self.estimate_gradient(objective_function, boards)
    self.logits += rate * gradient

  # the board is generated row by row, so only the last move can be a suicide
  # illegal moves are skipped, while captures are not prevented
  # because of that, the color distribution can be different than specified by self.logits
  def generate_board(self):
    board = Board(self.size)
    for y in range(self.size):
      for x in range(self.size):
        color = self.generate_color(x, y)
        if board.would_be_legal(color, board.loc(x, y)):
          board.play(color, board.loc(x, y))
    return board

  def generate_color(self, x, y):
    relative_probabilities = np.exp(self.logits[x, y])
    return random.choices(population = StochasticBoard.colors, weights = relative_probabilities)[0]

  def create_evaluation_table(self):
    table = []
    for _ in range(self.size):
      column = []
      for _ in range(self.size):
        column.append([[] for _ in StochasticBoard.colors])
      table.append(column)
    return table

  def add_board_to_evaluation_table(self, board, table, objective_function):
    evaluation = objective_function(board)
    for y in range(self.size):
      for x in range(self.size):
        color = board.board[board.loc(x, y)]
        table[x][y][color].append(evaluation)

  def average_evaluation_table(self, table, average_of_empty):
    averages = np.empty_like(self.logits)
    for y in range(self.size):
      for x in range(self.size):
        for color in StochasticBoard.colors:
          averages[x][y][color] = np.mean(table[x][y][color])
    return np.nan_to_num(averages, copy = False, nan = average_of_empty)

  def estimate_gradient(self, objective_function, boards):
    table = self.create_evaluation_table()
    for board in boards:
      self.add_board_to_evaluation_table(board, table, objective_function)
    overall_evaluation = np.mean(list(itertools.chain.from_iterable(table[0][0])))
    average_evaluations = self.average_evaluation_table(table, average_of_empty = overall_evaluation)
    return self.probabilities() * (average_evaluations - overall_evaluation)