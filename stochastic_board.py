import random
import scipy.stats
import numpy as np
from board import Board

class StochasticBoard:
  # we can index lists based on those without any conversion because they are equal to 0,1,2
  colors = [Board.EMPTY, Board.BLACK, Board.WHITE]
  assert(colors == [0, 1, 2])

  def __init__(self, size):
    self.size = size
    self.logits = np.zeros([size, size, len(StochasticBoard.colors)], dtype = np.float32)

  def entropies(self):
    return scipy.stats.entropy(np.exp(self.logits), axis=2)

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

  def generate_sample(self, sample_size):
    catalog = self.create_catalog()
    boards = []
    for _ in range(sample_size):
      board = self.generate_board()
      boards.append(board)
      self.add_board_to_catalog(board, catalog)
    return catalog, boards

  def create_catalog(self):
    catalog = []
    for _ in range(self.size):
      column = []
      for _ in range(self.size):
        column.append([[] for _ in StochasticBoard.colors])
      catalog.append(column)

  def add_board_to_catalog(self, board, catalog):
    for y in range(self.size):
      for x in range(self.size):
        color = board.board[board.loc(x, y)]
        catalog[x][y][color].append(board)