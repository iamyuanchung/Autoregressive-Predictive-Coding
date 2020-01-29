from collections import namedtuple


PrenetConfig = namedtuple(
  'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])

RNNConfig = namedtuple(
  'RNNConfig',
  ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])
