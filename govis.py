#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import re
import logging
import colorsys
import json
import tensorflow as tf
import numpy as np
from board import Board
from model import Model
import common

def main():
  model_variables_prefix = "nets/g170-b6c96-s175395328-d26788732/saved_model/variables/variables"
  model_config_path = "nets/g170-b6c96-s175395328-d26788732/model.config.json"
  name_scope = "swa_model"
  board_size = 19
  gs = GameState(board_size)
  rules = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5
  }
  model = make_model(name_scope, model_config_path)
  value_output = tf.nn.softmax(model.value_output)
  with tf.Session() as session:
    restore_session(session, model_variables_prefix)
    value = fetch_output(value_output, model, rules, gs, session)
  print(value)

def make_model(name_scope, config_path):
  max_board_size = 19
  with open(config_path) as f:
    config = json.load(f)
  with tf.compat.v1.variable_scope(name_scope):
    return Model(config, max_board_size, {})

def restore_session(session, model_variables_prefix):
  saver = tf.train.Saver(max_to_keep = 10000, save_relative_paths = True)
  saver.restore(session, model_variables_prefix)

def fetch_output(output, model, rules, gs, session):
  bin_input_data = np.zeros(shape = [1] + model.bin_input_shape, dtype = np.float32)
  global_input_data = np.zeros(shape = [1] + model.global_input_shape, dtype = np.float32)
  pla = gs.board.pla
  opp = Board.get_opp(pla)
  move_index = len(gs.moves)
  model.fill_row_features(gs.board, pla, opp, gs.boards, gs.moves, move_index, rules, bin_input_data, global_input_data, idx = 0)
  outputs = session.run([output], feed_dict = {
    model.bin_inputs: bin_input_data,
    model.global_inputs: global_input_data,
    model.symmetries: [False,False,False],
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  return outputs[0][0]

class GameState:
  def __init__(self,board_size):
    self.board_size = board_size
    self.board = Board(size = board_size)
    self.moves = []
    self.boards = [self.board.copy()]

main()