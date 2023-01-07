import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch import distributions as torchd

from utils_folder import utils
from utils_folder.utils import SquashedNormal
from utils_folder.utils_dreamer import Bernoulli


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, dist=None):
        super().__init__()

        self.dist = dist
        self._shape = (1,)
                
        output_dim = 1
        self.trunk = utils.mlp(input_dim, hidden_dim, output_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)
       
    def forward(self, transition):
        d = self.trunk(transition)
        if self.dist == 'binary':
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=d), len(self._shape)))
        else:
            return d 

class DDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.trunk = utils.mlp(obs_dim, hidden_dim, action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.trunk(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

class DDPG_Agent:
    """DDPG algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, hidden_dim,
                 hidden_depth, discount, actor_lr, actor_betas, actor_update_frequency, 
                 critic_lr, critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, num_expl_steps, stddev_schedule, stddev_clip,
                 reward_d_coef, discriminator_lr, GAN_loss='bce', from_dem=False):

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau

        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.GAN_loss = GAN_loss
        self.from_dem = from_dem

        self.critic = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DDPGActor(obs_dim, action_dim, hidden_dim, 
                                hidden_depth).to(self.device)

        # added model
        if from_dem:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(obs_dim + action_dim, hidden_dim, hidden_depth).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(obs_dim + action_dim, hidden_dim, hidden_depth, dist='binary').to(device)

            else:
                NotImplementedError

        else:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(2*obs_dim, hidden_dim, hidden_depth).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(2*obs_dim, hidden_dim, hidden_depth, dist='binary').to(device)

            else:
                NotImplementedError

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                                        lr=discriminator_lr)

        self.train()
        self.critic_target.train()
        
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(self.action_range[0], self.action_range[1])
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done *self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor(self, obs, logger, step):

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        logger.log('train_actor/loss', actor_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

    def compute_reward(self, obs_a, next_a, logger, step):
        
        with torch.no_grad():
            self.discriminator.eval()
            transition_a = torch.cat([obs_a, next_a], dim=-1)
            d = self.discriminator(transition_a)

            if self.GAN_loss == 'least-square':
                reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)

            elif self.GAN_loss == 'bce':
                reward_d = d.mode()

            reward = reward_d

            logger.log('train_discriminator/reward_d', reward_d.mean(), step)
            logger.log('train_discriminator/reward', reward.mean(), step)
            
            self.discriminator.train()
            
        return reward
    
    def compute_discriminator_grad_penalty_LS(self, obs_e, next_e, lambda_=10):
        
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        expert_data.requires_grad = True
        
        d = self.discriminator.trunk(expert_data)
        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=expert_data, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        
        return grad_pen

    def compute_discriminator_grad_penalty_bce(self, obs_a, next_a, obs_e, next_e, lambda_=10):

        agent_feat = torch.cat([obs_a, next_a], dim=-1)
        alpha = torch.rand(agent_feat.shape[:1]).unsqueeze(-1).to(self.device)
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        disc_penalty_input = alpha*agent_feat + (1-alpha)*expert_data

        disc_penalty_input.requires_grad = True
        d = self.discriminator(disc_penalty_input).mode()
        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=disc_penalty_input, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
        
    def update_discriminator(self, obs_a, next_a, obs_e, next_e, logger, step):
        
        transition_a = torch.cat([obs_a, next_a], dim=-1)
        transition_e = torch.cat([obs_e, next_e], dim=-1)
        
        agent_d = self.discriminator(transition_a)
        expert_d = self.discriminator(transition_e)

        if self.GAN_loss == 'least-square':
            expert_loss = F.mse_loss(expert_d, torch.ones(expert_d.size(), device=self.device))
            agent_loss = F.mse_loss(agent_d, -1*torch.ones(agent_d.size(), device=self.device))
            grad_pen_loss = self.compute_discriminator_grad_penalty_LS(obs_e, next_e)
            loss = 0.5*(expert_loss + agent_loss) + grad_pen_loss

        elif self.GAN_loss == 'bce':
            expert_loss = (expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self.device))).mean()
            agent_loss = (agent_d.log_prob(torch.zeros_like(agent_d.mode()).to(self.device))).mean()
            grad_pen_loss = self.compute_discriminator_grad_penalty_bce(obs_a.detach(), next_a.detach(), obs_e.detach(), next_e.detach())
            loss = -(expert_loss+agent_loss) + grad_pen_loss

        logger.log('train_discriminator/expert_loss', expert_loss, step)
        logger.log('train_discriminator/agent_loss', agent_loss, step)
        logger.log('train_discriminator/grad_pen_loss', grad_pen_loss, step)
        logger.log('train_discriminator/loss', loss, step)
        
        # optimize inverse models
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step() 

    def update(self, replay_buffer, replay_buffer_expert, logger, step):

        obs, action, reward_a, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
        obs_e, action_e, _, next_obs_e, _, _ = replay_buffer_expert.sample(self.batch_size)

        if self.from_dem:
            self.update_discriminator(obs, action, obs_e, action_e, logger, step)
            reward = self.compute_reward(obs, action, logger, step)
        else:
            self.update_discriminator(obs, next_obs, obs_e, next_obs_e, logger, step)
            reward = self.compute_reward(obs, next_obs, logger, step)

        logger.log('train/batch_reward_agent_only', reward_a.mean(), step)
        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
