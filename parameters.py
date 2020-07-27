import os
from input import FullInputBuilder, QuickInputBuilder
from board import Board

board_size = 19
katago_color = Board.BLACK
window_size = 800
InputBuilder = FullInputBuilder
network_path = 'nets/g170-b6c96-s175395328-d26788732'
model_parameters = {
  'variables_prefix': os.path.join(network_path, 'saved_model/variables/variables'),
  'config_path': os.path.join(network_path, 'model.config.json'),
  'name_scope': 'swa_model',
  'channel_size': 19,
}
neuron_location = {
  'layer': 0,
  'channel': 0,
  'x': 0,
  'y': 0
}
hyperparameters = {
  'iteration_count': 50,
  'rate': 0.5,
  'sample_size': 20
}
rules = {
  'koRule': 'KO_SIMPLE',
  'scoringRule': 'SCORING_AREA',
  'taxRule': 'TAX_NONE',
  'multiStoneSuicideLegal': True,
  'whiteKomi': 7.5
}
color_scheme = {
  'black_rgb': (0, 0, 0),
  'white_rgb': (255, 255, 255),
  'line_rgb': (0, 0, 0),
  'background_rgb': (250, 170, 80),
  'max_opacity': 255
}