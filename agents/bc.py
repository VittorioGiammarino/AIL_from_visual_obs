import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd
from torch import autograd
from torch.nn.utils import spectral_norm 

from utils_folder import utils
from utils_folder.utils_dreamer import Bernoulli

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.feature_dim = (32,35,35)

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        z = self.trunk(h)
        return z
        
class Actor(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class BcAgent:
    def __init__(self, 
                 obs_shape, 
                 action_shape, 
                 device, 
                 lr, 
                 feature_dim,
                 hidden_dim, 
                 actor_target_tau, 
                 stddev_clip,
                 use_tb):
        
        self.device = device
        self.actor_target_tau = actor_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip

        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.actor = Actor(action_shape, feature_dim, hidden_dim).to(device)
        self.actor_target = Actor(action_shape, feature_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.actor_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = 0.1
        dist = self.actor_target(obs, stddev)
        action = dist.mean

        return action.cpu().numpy()[0]

    def update_actor(self, obs_e, action_e):
        metrics = dict()

        obs_e = self.encoder(obs_e)
        dist = self.actor(obs_e, 0.1)
        recon = dist.sample(clip=self.stddev_clip)

        actor_loss = F.mse_loss(recon, action_e)

        # optimize actor
        self.encoder_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.encoder_opt.step()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics  

    def update(self, replay_iter_expert):
        metrics = dict()
        
        batch_expert = next(replay_iter_expert)
        obs_e_raw, action_e, _, _, _ = utils.to_torch(batch_expert, self.device)
        
        obs_e = self.aug(obs_e_raw.float())

        # update actor and encoder
        metrics.update(self.update_actor(obs_e, action_e))

        # update critic target
        utils.soft_update_params(self.actor, self.actor_target, self.actor_target_tau)

        return metrics
