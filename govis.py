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
from input import generate_input

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
  board = Board(board_size)
  model_config = read_config(model_config_path)
  model = make_model(name_scope, channel_size, model_config)
  value_output = tf.nn.softmax(model.value_output)
  channel_input, global_input = generate_input(model, board, Board.BLACK, channel_size, rules, model_config)
  with tf.Session() as session:
    restore_session(session, model_variables_prefix)
    outputs = session.run([value_output], feed_dict = {
      model.bin_inputs: channel_input,
      model.global_inputs: global_input,
      model.symmetries: [False,False,False],
      model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
    })
    print(outputs)
    # layer_name, layer = model.outputs_by_layer[3]
    # print(layer_name)
    # print(layer.shape)
    # print(session.run(layer))

def read_config(config_path):
  with open(config_path) as f:
    return json.load(f)

def make_model(name_scope, channel_size, config):
  with tf.compat.v1.variable_scope(name_scope):
    return Model(config, channel_size, {})

def restore_session(session, model_variables_prefix):
  saver = tf.train.Saver(max_to_keep = 10000, save_relative_paths = True)
  saver.restore(session, model_variables_prefix)

main()