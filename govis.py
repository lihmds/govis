import random
import json
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from board import Board
from model import Model
from input import InputBuilder, FractionalInputBuilder

def main():
  model_variables_prefix = "nets/g170-b6c96-s175395328-d26788732/saved_model/variables/variables"
  model_config_path = "nets/g170-b6c96-s175395328-d26788732/model.config.json"
  name_scope = "swa_model"
  rules = {
    "koRule": "KO_SIMPLE",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5
  }
  board_size = 19
  channel_size = 19
  model = make_model(name_scope, channel_size, model_config_path)
  winrate = get_winrate(model)
  with tf.Session() as session:
    restore_session(session, model_variables_prefix)
    def compute_winrate(board, truth_value):
      return apply_net_to_board(session, FractionalInputBuilder(truth_value), model, board, Board.BLACK, rules, winrate)
    plot_against_truth_value(compute_winrate, board_size, 5)

def plot_against_truth_value(f, board_size, plot_count):
  truth_values = np.linspace(0.0, 1.0)
  _, axes = plot.subplots()
  axes.set(xlabel = 'truth value', ylabel = 'output')
  axes.margins(0.1)
  axes.grid()
  for _ in range(plot_count):
    board = generate_board(board_size)
    outputs = list(map(lambda truth_value: f(board, truth_value), truth_values))
    axes.plot(truth_values, outputs)
  plot.show()

def get_winrate(model):
  value_output = tf.nn.softmax(model.value_output)
  return value_output[0,0]

def generate_board(size):
  board = Board(size)
  for y in range(board.size):
    for x in range(board.size):
      if random.random() < 0.05:
        player = random.choice([Board.BLACK, Board.WHITE])
        location = board.loc(x, y)
        if board.would_be_legal(player, location):
          board.play(player, location)
  return board

def apply_net_to_board(session, input_builder, model, board, own_color, rules, output):
  channel_input, global_input = input_builder.build(model, board, own_color, rules)
  return session.run([output], feed_dict = {
    model.bin_inputs: channel_input,
    model.global_inputs: global_input,
    model.symmetries: [False, False, False],
    model.include_history: [[1.0, 1.0, 1.0, 1.0, 1.0]]
  })

def make_model(name_scope, channel_size, config_path):
  with open(config_path) as f:
    config = json.load(f)
  with tf.compat.v1.variable_scope(name_scope):
    return Model(config, channel_size, {})

def restore_session(session, model_variables_prefix):
  saver = tf.train.Saver(max_to_keep = 10000, save_relative_paths = True)
  saver.restore(session, model_variables_prefix)

main()