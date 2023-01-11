import torch
from torch import nn
from torch import distributions as torchd

from utils_folder import utils_dreamer as utils

class Random(nn.Module):

  def __init__(self, config):
    self._config = config

  def actor(self, feat):
    shape = feat.shape[:-1] + [self._config.num_actions]
    if self._config.actor_dist == 'onehot':
      return utils.OneHotDist(torch.zeros(shape))
    else:
      ones = torch.ones(shape)
      return utils.ContDist(torchd.uniform.Uniform(-ones, ones))

  def train(self, start, context):
    return None, {}


