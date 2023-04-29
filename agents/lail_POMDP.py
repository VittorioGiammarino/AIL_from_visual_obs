# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch import distributions as torchd

from utils_folder import utils
from utils_folder.utils_dreamer import Bernoulli

class TruncatedGaussianPolicy(nn.Module):
    """
    Policy parameterized as diagonal gaussian distribution.
    """
    def __init__(self, action_shape, num_sequences, feature_dim, hidden_units=(256, 256)):
        super(TruncatedGaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = utils.build_mlp(
            input_dim = num_sequences * feature_dim + (num_sequences - 1) * action_shape,
            output_dim=action_shape,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(utils.initialize_weight)

    def forward(self, feature_action, std):
        means = self.net(feature_action)
        mu = torch.tanh(means)
        std = torch.ones_like(mu)*std 

        dist = utils.TruncatedNormal(mu, std)
        return dist

class TwinnedQNetwork(nn.Module):
    """
    Twinned Q networks.
    """
    def __init__(self, action_shape, z1_dim, z2_dim, hidden_units=(256, 256)):
        super(TwinnedQNetwork, self).__init__()

        self.net1 = utils.build_mlp(
            input_dim=action_shape + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(utils.initialize_weight)

        self.net2 = utils.build_mlp(
            input_dim=action_shape + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(utils.initialize_weight)

    #@torch.jit.script_method
    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        return self.net1(x), self.net2(x)

class FixedGaussian(nn.Module):  # torch.jit.ScriptModule
    """
    Fixed diagonal gaussian distribution.
    """
    def __init__(self, output_dim, std):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    #@torch.jit.script_method
    def forward(self, x):
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std

class Gaussian(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """
    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Gaussian, self).__init__()
        self.net = utils.build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(utils.initialize_weight)

    #@torch.jit.script_method
    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std

class LatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    """
    def __init__(
        self,
        state_shape,
        action_shape,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
    ):
        super(LatentModel, self).__init__()
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(z2_dim + action_shape, z1_dim, hidden_units)
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(z1_dim + z2_dim + action_shape, z2_dim, hidden_units)

        # q(z1(0) | feat(0))
        self.z1_posterior_init = Gaussian(state_shape, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(state_shape + z2_dim + action_shape, z1_dim, hidden_units)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(2 * z1_dim + 2 * z2_dim + action_shape, 1, hidden_units)

        # p(x(t) | z1(t), z2(t))
        self.decoder = Gaussian(z1_dim + z2_dim, state_shape, hidden_units)
        self.apply(utils.initialize_weight)

    #@torch.jit.script_method
    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return z1_mean_, z1_std_

    #@torch.jit.script_method
    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return z1_mean_, z1_std_, z1_, z2_

    #@torch.jit.script_method
    def calculate_loss(self, state_, action_, reward_, done_):
        # Calculate the sequence of features.
        feature_ = state_

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = utils.calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # # Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)

        state_mean_, state_std_ = self.decoder(z_)
        state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_state_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_image = -log_likelihood_state_.mean(dim=0).sum()

        # Prediction loss of rewards.
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()
        return loss_kld, loss_image, loss_reward
        
class Discriminator(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_units = (256, 256), dist = None):
        super(Discriminator, self).__init__()

        self.dist = dist
        self._shape = (1,)
                
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.net = utils.build_mlp(input_dim=feature_dim, output_dim=1, hidden_units=hidden_units,
                        hidden_activation=nn.LeakyReLU(0.2)).apply(utils.initialize_weight)
       
    def forward(self, transition):
        d = self.net(self.trunk(transition))
        if self.dist == 'binary':
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=d), len(self._shape)))
        else:
            return d 

class LailPOMDPAgent:
    def __init__(self, obs_shape, action_shape, device, gamma, batch_size_actor,
                batch_size_latent, num_sequences, lr_actor, lr_latent, z1_dim, z2_dim, hidden_units, feature_dim,
                critic_target_tau, update_every_steps, use_tb, num_expl_steps, reward_d_coef, lr_discriminator, 
                stddev_schedule, stddev_clip, GAN_loss='bce', from_dem=False):

        # Networks.
        self.actor = TruncatedGaussianPolicy(action_shape, num_sequences, obs_shape, hidden_units).to(device)
        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.latent = LatentModel(obs_shape, action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.GAN_loss = GAN_loss
        self.from_dem = from_dem

        utils.soft_update(self.critic_target, self.critic, 1.0)
        utils.grad_false(self.critic_target)

        # Optimizers.
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_actor)
        self.optim_latent = torch.optim.Adam(self.latent.parameters(), lr=lr_latent)

        print(f"Use {self.GAN_loss} for IL")

        # Discriminator.
        if from_dem:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(2*(z1_dim+z2_dim), feature_dim, hidden_units).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(2*(z1_dim+z2_dim), feature_dim, hidden_units, dist='binary').to(device) 

            else:
                NotImplementedError

        else:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator((num_sequences+1)*obs_shape, feature_dim, hidden_units).to(device) 
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator((num_sequences+1)*obs_shape, feature_dim, hidden_units, dist='binary').to(device) 

            else:
                NotImplementedError

        self.optim_discr = torch.optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)

        self.state_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.batch_size_actor = batch_size_actor
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = critic_target_tau
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        # Other options.
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.use_tb = use_tb

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.latent.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)

    def get_features(self, ob):
        state = torch.tensor(ob.state, dtype=torch.float, device=self.device).float()
        feature = state.view(1, -1)
        return feature

    def create_feature_actions(self, feature_, action_):
        N = feature_.size(0)
        # Flatten sequence of features.
        f = feature_[:, :-1].view(N, -1)
        n_f = feature_[:, 1:].view(N, -1)
        # Flatten sequence of actions.
        a = action_[:, :-1].view(N, -1)
        n_a = action_[:, 1:].view(N, -1)
        # Concatenate feature and action.
        fa = torch.cat([f, a], dim=-1)
        n_fa = torch.cat([n_f, n_a], dim=-1)
        return fa, n_fa

    def preprocess(self, ob):
        feature = self.get_features(ob)
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device)
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def act(self, ob, step, eval_mode):
        feature_action = self.preprocess(ob)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(feature_action, stddev)
        with torch.no_grad():
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()[0]

    def update_latent(self, replay_iter):
        metrics = dict()

        state_, action_, reward_, done_ = replay_iter.sample_latent(self.batch_size_latent)
        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (loss_kld + loss_image + loss_reward).backward() 
        self.optim_latent.step()

        if self.use_tb:
            metrics["kld_loss"] = loss_kld.item()
            metrics["reward_loss"] = loss_reward.item()
            metrics["image_loss"] = loss_image.item()

        return metrics

    def prepare_batch(self, state_, action_):
        with torch.no_grad():
            # f(1:t+1)
            feature_ = state_
            # z(1:t+1)
            _, _, z1_, z2_ = self.latent.sample_posterior(feature_, action_)
            z_ = torch.cat([z1_, z2_], dim=-1)

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_feature_action, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = torch.min(next_q1, next_q2)

        target_q = reward + (1.0 - done) * self.gamma * next_q
        curr_q1, curr_q2 = self.critic(z, action)
        loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

        if self.use_tb:
            metrics["critic_loss"] = loss_critic.item()
            metrics["critic_q1"] = curr_q1.mean().item()
            metrics["critic_q2"] = curr_q2.mean().item()
            metrics["critic_target_q"] = target_q.mean().item()

        return metrics

    def update_actor(self, z, feature_action, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(feature_action, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        q1, q2 = self.critic(z, action)
        loss_actor = -torch.mean(torch.min(q1, q2))

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        if self.use_tb:
            metrics["actor_loss"] = loss_actor.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def compute_reward(self, feature_actor):
        with torch.no_grad():
            self.discriminator.eval()
            d = self.discriminator(feature_actor)

            if self.GAN_loss == 'least-square':
                reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            elif self.GAN_loss == 'bce':
                reward_d = d.mode()

            reward = reward_d

            self.discriminator.train()
            
        return reward
    
    def compute_discriminator_grad_penalty_LS(self, feature_expert, lambda_=10):
        feature_expert.requires_grad = True
        
        d = self.discriminator.net(self.discriminator.trunk(feature_expert))
        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=feature_expert, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def compute_discriminator_grad_penalty_bce(self, feature_actor, feature_expert, lambda_=10):

        alpha = torch.rand(feature_actor.shape[:1]).unsqueeze(-1).to(self.device)
        disc_penalty_input = alpha*feature_actor + (1-alpha)*feature_expert

        disc_penalty_input.requires_grad = True
        
        d = self.discriminator.net(self.discriminator.trunk(feature_expert))
        d = self.discriminator(disc_penalty_input).mode()
        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=disc_penalty_input, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
     
    def update_discriminator(self, feature_actor, feature_expert):
        metrics = dict()
        
        agent_d = self.discriminator(feature_actor)
        expert_d = self.discriminator(feature_expert)

        if self.GAN_loss == 'least-square':
            expert_labels = 1.0
            agent_labels = -1.0

            expert_loss = F.mse_loss(expert_d, expert_labels*torch.ones(expert_d.size(), device=self.device))
            agent_loss = F.mse_loss(agent_d, agent_labels*torch.ones(agent_d.size(), device=self.device))
            grad_pen_loss = self.compute_discriminator_grad_penalty_LS(feature_expert.detach())
            loss = 0.5*(expert_loss + agent_loss) + grad_pen_loss

        elif self.GAN_loss == 'bce':
            expert_loss = (expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self.device))).mean()
            agent_loss = (agent_d.log_prob(torch.zeros_like(agent_d.mode()).to(self.device))).mean()
            grad_pen_loss = self.compute_discriminator_grad_penalty_bce(feature_actor.detach(), feature_expert.detach())
            loss = -(expert_loss+agent_loss) + grad_pen_loss
        
        # optimize discriminator
        self.optim_discr.zero_grad(set_to_none=True)
        loss.backward()
        self.optim_discr.step()
        
        if self.use_tb:
            metrics['discriminator_expert_loss'] = expert_loss.item()
            metrics['discriminator_agent_loss'] = agent_loss.item()
            metrics['discriminator_loss'] = loss.item()
            metrics['discriminator_grad_pen_loss'] = grad_pen_loss.item()
        
        return metrics 

    def update(self, replay_iter, replay_iter_expert, step):
        metrics = dict()
        metrics.update(self.update_latent(replay_iter))
        
        if step % self.update_every_steps != 0:
            return metrics
    
        state_, action_, reward_a, done = replay_iter.sample_actor(self.batch_size_actor)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, action_)

        if self.from_dem:
            state_expert_, action_expert_, _, _ = replay_iter_expert.sample(self.batch_size_actor)
            z_expert, next_z_expert, _, _, _ = self.prepare_batch(state_expert_, action_expert_)
            feature_expert = torch.cat([z_expert, next_z_expert], dim=-1)
            feature_actor = torch.cat([z, next_z], dim=-1)

        else:
            state_expert_ = replay_iter_expert.sample_states_only(self.batch_size_actor)
            with torch.no_grad():
                feature_actor = state_.view(self.batch_size_actor, -1)
                feature_expert = state_expert_.view(self.batch_size_actor, -1)

        metrics.update(self.update_discriminator(feature_actor, feature_expert))
        reward = self.compute_reward(feature_actor)

        if self.use_tb:
            metrics['batch_reward_env'] = reward_a.mean().item()
            metrics['batch_reward'] = reward.mean().item()
                
        # update critic
        metrics.update(self.update_critic(z, next_z, action, next_feature_action, reward, done, step))

        # update actor
        metrics.update(self.update_actor(z, feature_action, step))

        # update critic target
        utils.soft_update(self.critic_target, self.critic, self.tau)

        return metrics