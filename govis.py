import json
import numpy as np
import tensorflow as tf
import pyglet
from graphics import BoardProbabilityDisplay
from parameters import board_size, katago_color, InputBuilder, model_parameters, neuron_location, hyperparameters, rules
from board import Board
from model import Model
from stochastic_board import StochasticBoard

def main():
  np.seterr(all = 'raise')
  stochastic_board = StochasticBoard(board_size)
  model = make_model()
  neuron = get_neuron(model)
  input_builder = InputBuilder(model)
  window_size = 800
  window = pyglet.window.Window(window_size, window_size)
  display = BoardProbabilityDisplay(board_size, window_size)
  with tf.compat.v1.Session() as session:
    restore_session(session)
    def objective_function(board):
      return apply_net_to_board(session, neuron, model, input_builder, board)
    for _ in range(hyperparameters['iteration_count']):
      stochastic_board.ascend_gradient(objective_function, hyperparameters['rate'], hyperparameters['sample_size'])
      window.dispatch_events()
      display.draw(stochastic_board.probabilities())
      window.flip()
  print(stochastic_board.probabilities())
  input("Press enter to close...")

def make_model():
  with open(model_parameters['config_path']) as f:
    config = json.load(f)
  with tf.compat.v1.variable_scope(model_parameters['name_scope']):
    return Model(config, model_parameters['channel_size'], {})

def get_neuron(model):
  layer_name, layer = model.outputs_by_layer[neuron_location['layer']]
  print('layer name:', layer_name)
  print('layer shape:', layer.shape)
  return layer[0, neuron_location['y'], neuron_location['x'], neuron_location['channel']]

def restore_session(session):
  saver = tf.compat.v1.train.Saver()
  saver.restore(session, model_parameters['variables_prefix'])

def apply_net_to_board(session, output, model, input_builder, board):
  return session.run(output, feed_dict = {
    model.bin_inputs: input_builder.build_channels(board, katago_color, rules),
    model.global_inputs: input_builder.build_globals(board, katago_color, rules),
    model.symmetries: [False, False, False],
    model.include_history: [[0.0, 0.0, 0.0, 0.0, 0.0]]
  })

main()