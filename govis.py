import random
import json
import os
import numpy as np
import tensorflow as tf
from board import Board
from model import Model
from input import InputBuilder, QuickInputBuilder
from stochastic_board import StochasticBoard

def main():
  np.seterr(all = 'raise')
  network_path = "nets/g170-b6c96-s175395328-d26788732"
  model_variables_prefix = os.path.join(network_path, "saved_model/variables/variables")
  model_config_path = os.path.join(network_path, "model.config.json")
  name_scope = "swa_model"
  rules = {
    "koRule": "KO_SIMPLE",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "whiteKomi": 7.5
  }
  channel_size = 19
  stochastic_board = StochasticBoard(19)
  model = make_model(name_scope, channel_size, model_config_path)
  neuron = get_some_neuron(model)

  with tf.compat.v1.Session() as session:
    restore_session(session, model_variables_prefix)
    def objective_function(board):
      return apply_net_to_board(session, InputBuilder(model), model, board, Board.BLACK, rules, neuron)
    for _ in range(100):
      stochastic_board.ascend_gradient(objective_function, 0.5, 20)
      print(stochastic_board.generate_board().to_string(), '\n\n')
    print(stochastic_board.entropies())

def get_some_neuron(model):
  layer_name, layer = model.outputs_by_layer[0]
  print('layer name:', layer_name)
  return layer[0, 0, 0, 0]

def apply_net_to_board(session, input_builder, model, board, own_color, rules, output):
  return session.run(output, feed_dict = {
    model.bin_inputs: input_builder.build_channels(board, own_color, rules),
    model.global_inputs: input_builder.build_globals(board, own_color, rules),
    model.symmetries: [False, False, False],
    model.include_history: [[0.0, 0.0, 0.0, 0.0, 0.0]]
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