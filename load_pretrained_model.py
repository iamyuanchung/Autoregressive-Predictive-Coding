"""Example of loading a pre-trained APC model."""

import torch

from apc_model import APCModel
from utils import PrenetConfig, RNNConfig


def main():
  prenet_config = None
  rnn_config = RNNConfig(input_size=80, hidden_size=512, num_layers=3,
                         dropout=0.)
  pretrained_apc = APCModel(mel_dim=80, prenet_config=prenet_config,
                            rnn_config=rnn_config).cuda()

  pretrained_weights_path = 'bs32-rhl3-rhs512-rd0-adam-res-ts3.pt'
  pretrained_apc.load_state_dict(torch.load(pretrained_weights_path))

  # Load data and perform your task ...
