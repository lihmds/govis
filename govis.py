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

description = """
Play go with a trained neural net!
Implements a basic GTP engine that uses the neural net directly to play moves.
"""

parser = argparse.ArgumentParser(description=description)
common.add_model_load_args(parser)
parser.add_argument('-name-scope', help='Name scope for model variables', required=False)
args = vars(parser.parse_args())

(model_variables_prefix, model_config_json) = common.load_model_paths(args)
name_scope = args["name_scope"]

#Hardcoded max board size
pos_len = 19

# Model ----------------------------------------------------------------

with open(model_config_json) as f:
  model_config = json.load(f)

if name_scope is not None:
  with tf.compat.v1.variable_scope(name_scope):
    model = Model(model_config,pos_len,{})
else:
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


# Moves ----------------------------------------------------------------

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


def get_gfx_commands_for_heatmap(locs_and_values, board, normalization_div, is_percent, value_and_score_from=None, hotcold=False):
  gfx_commands = []
  divisor = 1.0
  if normalization_div == "max":
    max_abs_value = max(abs(value) for (loc,value) in locs_and_values)
    divisor = max(0.0000000001,max_abs_value) #avoid divide by zero
  elif normalization_div is not None:
    divisor = normalization_div

  #Caps value at 1.0, using an asymptotic curve
  def loose_cap(x):
    def transformed_softplus(x):
      return -math.log(math.exp(-(x-1.0)*8.0)+1.0)/8.0+1.0
    base = transformed_softplus(0.0)
    return (transformed_softplus(x) - base) / (1.0 - base)

  #Softly curves a value so that it ramps up faster than linear in that range
  def soft_curve(x,x0,x1):
    p = (x-x0)/(x1-x0)
    def curve(p):
      return math.sqrt(p+0.16)-0.4
    p = curve(p) / curve(1.0)
    return x0 + p * (x1-x0)

  if hotcold:
    for (loc,value) in locs_and_values:
      if loc != Board.PASS_LOC:
        value = value / divisor

        if value < 0:
          value = -loose_cap(-value)
        else:
          value = loose_cap(value)

        interpoints = [
          (-1.00,(0,0,0)),
          (-0.85,(15,0,50)),
          (-0.60,(60,0,160)),
          (-0.35,(0,0,255)),
          (-0.15,(0,100,255)),
          ( 0.00,(115,115,115)),
          ( 0.15,(250,45,40)),
          ( 0.25,(255,55,0)),
          ( 0.60,(255,255,20)),
          ( 0.85,(255,255,128)),
          ( 1.00,(255,255,255)),
        ]

        def lerp(p,y0,y1):
          return y0 + p*(y1-y0)

        i = 0
        while i < len(interpoints):
          if value <= interpoints[i][0]:
            break
          i += 1
        i -= 1

        if i < 0:
          (r,g,b) = interpoints[0][1]
        if i >= len(interpoints)-1:
          (r,g,b) = interpoints[len(interpoints)-1][1]

        p = (value - interpoints[i][0]) / (interpoints[i+1][0] - interpoints[i][0])

        (r0,g0,b0) = interpoints[i][1]
        (r1,g1,b1) = interpoints[i+1][1]
        r = lerp(p,r0,r1)
        g = lerp(p,g0,g1)
        b = lerp(p,b0,b1)

        r = ("%02x" % int(r))
        g = ("%02x" % int(g))
        b = ("%02x" % int(b))
        gfx_commands.append("COLOR #%s%s%s %s" % (r,g,b,str_coord(loc,board)))

  else:
    for (loc,value) in locs_and_values:
      if loc != Board.PASS_LOC:
        value = value / divisor
        if value < 0:
          value = -value
          huestart = 0.50
          huestop = 0.86
        else:
          huestart = -0.02
          huestop = 0.38

        value = loose_cap(value)

        def lerp(p,x0,x1,y0,y1):
          return y0 + (y1-y0) * (p-x0)/(x1-x0)

        if value <= 0.03:
          hue = huestart
          lightness = 0.00 + 0.50 * (value / 0.03)
          saturation = value / 0.03
          (r,g,b) = colorsys.hls_to_rgb((hue+1)%1, lightness, saturation)
        elif value <= 0.60:
          hue = lerp(value,0.03,0.60,huestart,huestop)
          val = 1.0
          saturation = 1.0
          (r,g,b) = colorsys.hsv_to_rgb((hue+1)%1, val, saturation)
        else:
          hue = huestop
          lightness = lerp(value,0.60,1.00,0.5,0.95)
          saturation = 1.0
          (r,g,b) = colorsys.hls_to_rgb((hue+1)%1, lightness, saturation)

        r = ("%02x" % int(r*255))
        g = ("%02x" % int(g*255))
        b = ("%02x" % int(b*255))
        gfx_commands.append("COLOR #%s%s%s %s" % (r,g,b,str_coord(loc,board)))

  locs_and_values = sorted(locs_and_values, key=lambda loc_and_value: loc_and_value[1])
  locs_and_values_rev = sorted(locs_and_values, key=lambda loc_and_value: loc_and_value[1], reverse=True)
  texts = []
  texts_rev = []
  texts_value = []
  maxlen_per_side = 1000
  if len(locs_and_values) > 0 and locs_and_values[0][1] < 0:
    maxlen_per_side = 500

    for i in range(min(len(locs_and_values),maxlen_per_side)):
      (loc,value) = locs_and_values[i]
      if is_percent:
        texts.append("%s %4.1f%%" % (str_coord(loc,board),value*100))
      else:
        texts.append("%s %.3f" % (str_coord(loc,board),value))
    texts.reverse()

  for i in range(min(len(locs_and_values_rev),maxlen_per_side)):
    (loc,value) = locs_and_values_rev[i]
    if is_percent:
      texts_rev.append("%s %4.1f%%" % (str_coord(loc,board),value*100))
    else:
      texts_rev.append("%s %.3f" % (str_coord(loc,board),value))

  if value_and_score_from is not None:
    value = value_and_score_from["value"]
    score = value_and_score_from["scoremean"]
    lead = value_and_score_from["lead"]
    vtime = value_and_score_from["vtime"]
    texts_value.append("wv %.2fc nr %.2f%% ws %.1f wl %.1f vt %.1f" % (
      100*(value[0]-value[1] if board.pla == Board.WHITE else value[1] - value[0]),
      100*value[2],
      (score if board.pla == Board.WHITE else -score),
      (lead if board.pla == Board.WHITE else -lead),
      vtime
    ))

  gfx_commands.append("TEXT " + ", ".join(texts_value + texts_rev + texts))
  return gfx_commands