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

# Hardcoded max board size
pos_len = 19

with open(model_config_json) as f:
  model_config = json.load(f)

with tf.compat.v1.variable_scope(name_scope):
  model = Model(model_config,pos_len,{})

policy0_output = tf.nn.softmax(model.policy_output[:,:,0])
policy1_output = tf.nn.softmax(model.policy_output[:,:,1])
value_output = tf.nn.softmax(model.value_output)
scoremean_output = 20.0 * model.miscvalues_output[:,0]
scorestdev_output = 20.0 * tf.math.softplus(model.miscvalues_output[:,1])
lead_output = 20.0 * model.miscvalues_output[:,2]
vtime_output = 150.0 * tf.math.softplus(model.miscvalues_output[:,3])
ownership_output = tf.tanh(model.ownership_output)
scoring_output = model.scoring_output
futurepos_output = tf.tanh(model.futurepos_output)
seki_output = tf.nn.softmax(model.seki_output[:,:,:,0:3])
seki_output = seki_output[:,:,:,1] - seki_output[:,:,:,2]
seki_output2 = tf.sigmoid(model.seki_output[:,:,:,3])
scorebelief_output = tf.nn.softmax(model.scorebelief_output)
sbscale_output = model.sbscale3_layer

class GameState:
  def __init__(self,board_size):
    self.board_size = board_size
    self.board = Board(size=board_size)
    self.moves = []
    self.boards = [self.board.copy()]

def fetch_output(session, gs, rules, fetches):
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = gs.board.pla
  opp = Board.get_opp(pla)
  move_idx = len(gs.moves)
  model.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)
  outputs = session.run(fetches, feed_dict={
    model.bin_inputs: bin_input_data,
    model.global_inputs: global_input_data,
    model.symmetries: [False,False,False],
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  return [output[0] for output in outputs]

def get_outputs(session, gs, rules):
  [policy0,
   policy1,
   value,
   scoremean,
   scorestdev,
   lead,
   vtime,
   ownership,
   scoring,
   futurepos,
   seki,
   seki2,
   scorebelief,
   sbscale
  ] = fetch_output(session,gs,rules,[
    policy0_output,
    policy1_output,
    value_output,
    scoremean_output,
    scorestdev_output,
    lead_output,
    vtime_output,
    ownership_output,
    scoring_output,
    futurepos_output,
    seki_output,
    seki_output2,
    scorebelief_output,
    sbscale_output
  ])
  board = gs.board

  moves_and_probs0 = []
  for i in range(len(policy0)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy0)-1:
      moves_and_probs0.append((Board.PASS_LOC,policy0[i]))
    elif board.would_be_legal(board.pla,move):
      moves_and_probs0.append((move,policy0[i]))

  moves_and_probs1 = []
  for i in range(len(policy1)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy1)-1:
      moves_and_probs1.append((Board.PASS_LOC,policy1[i]))
    elif board.would_be_legal(board.pla,move):
      moves_and_probs1.append((move,policy1[i]))

  ownership_flat = ownership.reshape([model.pos_len * model.pos_len])
  ownership_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        ownership_by_loc.append((loc,ownership_flat[pos]))
      else:
        ownership_by_loc.append((loc,-ownership_flat[pos]))

  scoring_flat = scoring.reshape([model.pos_len * model.pos_len])
  scoring_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        scoring_by_loc.append((loc,scoring_flat[pos]))
      else:
        scoring_by_loc.append((loc,-scoring_flat[pos]))

  futurepos0_flat = futurepos[:,:,0].reshape([model.pos_len * model.pos_len])
  futurepos0_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        futurepos0_by_loc.append((loc,futurepos0_flat[pos]))
      else:
        futurepos0_by_loc.append((loc,-futurepos0_flat[pos]))

  futurepos1_flat = futurepos[:,:,1].reshape([model.pos_len * model.pos_len])
  futurepos1_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        futurepos1_by_loc.append((loc,futurepos1_flat[pos]))
      else:
        futurepos1_by_loc.append((loc,-futurepos1_flat[pos]))

  seki_flat = seki.reshape([model.pos_len * model.pos_len])
  seki_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        seki_by_loc.append((loc,seki_flat[pos]))
      else:
        seki_by_loc.append((loc,-seki_flat[pos]))

  seki_flat2 = seki2.reshape([model.pos_len * model.pos_len])
  seki_by_loc2 = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      seki_by_loc2.append((loc,seki_flat2[pos]))

  moves_and_probs = sorted(moves_and_probs0, key=lambda moveandprob: moveandprob[1], reverse=True)
  #Generate a random number biased small and then find the appropriate move to make
  #Interpolate from moving uniformly to choosing from the triangular distribution
  alpha = 1
  beta = 1 + math.sqrt(max(0,len(gs.moves)-20))
  r = np.random.beta(alpha,beta)
  probsum = 0.0
  i = 0
  genmove_result = Board.PASS_LOC
  while True:
    (move,prob) = moves_and_probs[i]
    probsum += prob
    if i >= len(moves_and_probs)-1 or probsum > r:
      genmove_result = move
      break
    i += 1

  return {
    "policy0": policy0,
    "policy1": policy1,
    "moves_and_probs0": moves_and_probs0,
    "moves_and_probs1": moves_and_probs1,
    "value": value,
    "scoremean": scoremean,
    "scorestdev": scorestdev,
    "lead": lead,
    "vtime": vtime,
    "ownership": ownership,
    "ownership_by_loc": ownership_by_loc,
    "scoring": scoring,
    "scoring_by_loc": scoring_by_loc,
    "futurepos": futurepos,
    "futurepos0_by_loc": futurepos0_by_loc,
    "futurepos1_by_loc": futurepos1_by_loc,
    "seki": seki,
    "seki_by_loc": seki_by_loc,
    "seki2": seki2,
    "seki_by_loc2": seki_by_loc2,
    "scorebelief": scorebelief,
    "sbscale": sbscale,
    "genmove_result": genmove_result
  }

def get_layer_values(session, gs, rules, layer, channel):
  board = gs.board
  [layer] = fetch_output(session,gs,rules=rules,fetches=[layer])
  layer = layer.reshape([model.pos_len * model.pos_len,-1])
  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      locs_and_values.append((loc,layer[pos,channel]))
  return locs_and_values

def get_input_feature(gs, rules, feature_idx):
  board = gs.board
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = board.pla
  opp = Board.get_opp(pla)
  move_idx = len(gs.moves)
  model.fill_row_features(board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)

  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      locs_and_values.append((loc,bin_input_data[0,pos,feature_idx]))
  return locs_and_values

def get_pass_alive(board, rules):
  pla = board.pla
  opp = Board.get_opp(pla)
  area = [-1 for i in range(board.arrsize)]
  nonPassAliveStones = False
  safeBigTerritories = True
  unsafeBigTerritories = False
  board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules["multiStoneSuicideLegal"])

  locs_and_values = []
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      locs_and_values.append((loc,area[loc]))
  return locs_and_values

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