import random
import json
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
    "whiteKomi": 0.5
  }
  channel_size = 19
  board = generate_board(19)
  model = make_model(name_scope, channel_size, model_config_path)
  # model.outputs_by_layer contains alternatives to value_output
  value_output = tf.nn.softmax(model.value_output)

  with tf.Session() as session:
    restore_session(session, model_variables_prefix)
    print('1.00 says:')
    print(apply_net_to_board(session, InputBuilder(), model, board, Board.BLACK, channel_size, rules, value_output))
    print('0.90 says:')
    print(apply_net_to_board(session, FractionalInputBuilder(0.9), model, board, Board.BLACK, channel_size, rules, value_output))

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

def apply_net_to_board(session, input_builder, model, board, own_color, channel_size, rules, output):
  channel_input, global_input = input_builder.build(model, board, own_color, channel_size, rules)
  return apply_net_to_inputs(session, model, channel_input, global_input, output)

def apply_net_to_inputs(session, model, channel_input, global_input, output):
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