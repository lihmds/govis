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

model_variables_prefix = "nets/g170-b6c96-s175395328-d26788732/saved_model/variables/variables"
model_config_json = "nets/g170-b6c96-s175395328-d26788732/model.config.json"
name_scope = "swa_model"

pos_len = 19 # Hardcoded max board size

with open(model_config_json) as f:
  model_config = json.load(f)

with tf.compat.v1.variable_scope(name_scope):
  model = Model(model_config,pos_len,{})

value_output = tf.nn.softmax(model.value_output)

class GameState:
  def __init__(self,board_size):
    self.board_size = board_size
    self.board = Board(size=board_size)
    self.moves = []
    self.boards = [self.board.copy()]

def main(session):
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
  fetches = [value_output]
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = gs.board.pla
  opp = Board.get_opp(pla)
  move_index = len(gs.moves)
  model.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_index,rules,bin_input_data,global_input_data,idx=0)
  outputs = session.run(fetches, feed_dict={
    model.bin_inputs: bin_input_data,
    model.global_inputs: global_input_data,
    model.symmetries: [False,False,False],
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  [[value]] = outputs
  print(value)

saver = tf.train.Saver(max_to_keep = 10000, save_relative_paths = True)
with tf.Session() as session:
  saver.restore(session, model_variables_prefix)
  main(session)