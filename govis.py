import random
import json
import numpy as np
import tensorflow as tf
from board import Board
from model import Model
from input import InputBuilder
from stochastic_board import StochasticBoard

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
  channel_size = 19
  model = make_model(name_scope, channel_size, model_config_path)
  layer_name, layer = model.outputs_by_layer[0]
  print(layer_name)
  neuron = layer[0, 0, 0, 0]
  stochastic_board = StochasticBoard(19)

  with tf.compat.v1.Session() as session:
    restore_session(session, model_variables_prefix)
    def objective_function(board):
      return apply_net_to_board(session, InputBuilder(), model, board, Board.BLACK, rules, neuron)
    for _ in range(100):
      stochastic_board.ascend_gradient(objective_function, 1.0, 20)
      print(stochastic_board.generate_board().to_string())
      print('\n')
    print(stochastic_board.entropies())

def apply_net_to_board(session, input_builder, model, board, own_color, rules, output):
  channel_input, global_input = input_builder.build(model, board, own_color, rules)
  return session.run(output, feed_dict = {
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
  saver = tf.compat.v1.train.Saver(max_to_keep = 10000, save_relative_paths = True)
  saver.restore(session, model_variables_prefix)

main()