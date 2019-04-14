from os import listdir
from os.path import join
import pickle

import torch
from torch.utils import data


class LibriSpeech(data.Dataset):
  def __init__(self, path):
    self.path = path
    self.ids = [f for f in listdir(self.path) if f.endswith('.pt')]
    with open(join(path, 'lengths.pkl'), 'rb') as f:
      self.lengths = pickle.load(f)

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, index):
    x = torch.load(join(self.path, self.ids[index]))
    l = self.lengths[self.ids[index]]
    return x, l
