
import hydra
import numpy as np
from torch import autograd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_folder import utils

def compute_gradient_penalty(discriminator, expert_data, policy_data, grad_pen_weight=10.0):
    if len(expert_data.shape) == 2:
        alpha = torch.rand(expert_data.size(0), 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)
    elif len(expert_data.shape) == 4:
        alpha = torch.rand(expert_data.size(0), 1, 1, 1, device=expert_data.device)

    mixup_data = alpha * expert_data + (1 - alpha) * policy_data
    mixup_data.requires_grad = True

    disc = discriminator(mixup_data)
    ones = torch.ones(disc.size()).to(disc.device)
    if len(expert_data.shape) == 2:
        grad = autograd.grad(outputs=disc,
                            inputs=mixup_data,
                            grad_outputs=ones,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
    elif len(expert_data.shape) == 4:
        grads = autograd.grad(outputs=disc.sum(),
                            inputs=mixup_data,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
        grad = grads.view(len(grads[0]), -1)

    grad_pen = grad_pen_weight * (grad.norm(2, dim=1) - 1).pow(2).sum()
    return grad_pen

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

class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    """Ref to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L538"""
    def __init__(self, in_dim, final_iid=False):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        sequence = [nn.Conv2d(in_dim, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(32, 64, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(64, 128, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, True), nn.Conv2d(128, 1, 4, stride=1, padding=1)]

        if final_iid:
            sequence += [nn.LeakyReLU(0.2, True), nn.Conv2d(1, 1, 1, stride=1, padding=0)]

        self.convnet = nn.Sequential(*sequence)
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.feature_dim = (32,35,35)

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class PatchAilAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, 
                 use_tb, reward_type="airl", sim_type="weight", reward_scale=1.0, grad_pen_weight=10.0, 
                 disc_lr=None, use_simreg=False, sim_rate=1.5):
        
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.sim_rate = sim_rate
        self.use_simreg = use_simreg
        self.sim_type = sim_type
        if disc_lr is None:
            disc_lr = lr
        if use_simreg:
            print("\nUsing Sim Reg, Sim Rate: {}".format(sim_rate))


        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.grad_pen_weight = grad_pen_weight

        assert reward_type in ["airl", "gail", "fairl", "gail2"], "Invalid adversarial irl reward type!"

        assert sim_type in ["weight", "bonus"], "Invalid sim type!"

        print("Using reward scale: {}\n".format(reward_scale))
        print("Using mean as reward aggregation")

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.encoder_target = Encoder(obs_shape).to(device)
        repr_dim = self.encoder.repr_dim
            
        disc_dim = 2*obs_shape[0]
        self.discriminator = PatchDiscriminator(disc_dim).to(device)
        self.actor = Actor(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=disc_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.disc_aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0)) 
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, std=stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                try:
                    action.uniform_(-1.0, 1.0)
                except:
                    action = dist.uniform()
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)

            dist = self.actor(next_obs, std=stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, std=stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            
        return metrics
    
    def update_discriminator(self, policy_obs, expert_obs, policy_next_obs=None, expert_next_obs=None):
        metrics = dict()
        batch_size = expert_obs.shape[0]
        obs_shape = expert_obs.shape[1]
        # policy batch size is 2x
        policy_obs = policy_obs[:batch_size]
        policy_next_obs = policy_next_obs[:batch_size]

        ones = torch.ones(batch_size, device=self.device)
        zeros = torch.zeros(batch_size, device=self.device)

        disc_obs = torch.cat([expert_obs, policy_obs], dim=0)

        # D(s,s')
        disc_next_obs = torch.cat([expert_next_obs, policy_next_obs], dim=0)
        disc_input = torch.cat([disc_obs, disc_next_obs], dim=1) # This is for PatchIRL

        disc_label = torch.cat([ones, zeros], dim=0).unsqueeze(dim=1)
        disc_output = self.discriminator(disc_input)

        patch_number = 1
        if disc_label.shape != disc_output.shape: # this is for patch gail - (B, P_W, P_H, 1)
            disc_output = disc_output.view(disc_output.shape[0],-1)
            patch_number = disc_output.shape[1]
            disc_label = disc_label.expand_as(disc_output)

        dac_loss = F.binary_cross_entropy_with_logits(disc_output, disc_label, reduction='sum')

        expert_obs, policy_obs = torch.split(disc_input, batch_size, dim=0)
        grad_pen = compute_gradient_penalty(self.discriminator, expert_obs.detach(), policy_obs.detach(), self.grad_pen_weight)

        dac_loss /= (batch_size * patch_number)
        grad_pen /= (batch_size * patch_number)

        metrics['disc_loss'] = dac_loss.mean().item()
        metrics['disc_grad_pen'] = grad_pen.mean().item()

        self.discriminator_opt.zero_grad(set_to_none=True)
        dac_loss.backward()
        grad_pen.backward()
        self.discriminator_opt.step()
        return metrics
    
    def record_grad_norm(self, model, net_name):
        """
        Record the grad norm for monitoring.
        """
        metrics = dict()
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        metrics[net_name+"grad_norm"] = total_norm

        return metrics

    def dac_rewarder(self, obses, next_obses, return_logits=False, clip=False):

        obses = torch.cat([obses, next_obses], dim=1)

        with torch.no_grad():
            with utils.eval_mode(self.discriminator):
                d = logits = self.discriminator(obses)
                if return_logits:
                    return logits

            d = d.mean(dim=(2,3), keepdim=True)

            s = torch.sigmoid(d)
            if self.reward_type == "airl": # If you compute log(D) - log(1-D) then you just get the logits
                reward = d # s.log() - (1 - s).log()
            elif self.reward_type == "gail":
                reward = - (1 - s).log()
            elif self.reward_type == "gail2":
                reward = s.log()
            elif self.reward_type == "fairl":
                reward = torch.exp(d) * (-1.0 * d)
            else:
                raise NotImplementedError
            
            if clip:
                reward = torch.clamp(reward, min=-10, max=10)

            reward = reward.sum(dim=(2,3))

        return self.reward_scale * reward
    
    def compute_similarity(self, obs_before_aug, expert_obs_before_aug, next_obs_before_aug, expert_next_obs_before_aug):

        # Compute the distance of the patch matrics between agent and expert
        similarity = 1
        expert_disc_input = torch.cat([expert_obs_before_aug, expert_next_obs_before_aug], dim=1)
        disc_input = torch.cat([obs_before_aug, next_obs_before_aug], dim=1) # use before aug obs for simreg

        expert_dist = torch.sigmoid(self.discriminator(expert_disc_input).detach().view(expert_disc_input.shape[0],-1))
        expert_dist = expert_dist.mean(dim=0, keepdim=True) # if use Eq(6), remove this line and change line 551 to line 550
        expert_dist /= expert_dist.sum(dim=1, keepdim=True)
        agent_dist = torch.sigmoid(self.discriminator(disc_input).detach().view(disc_input.shape[0],-1))
        agent_dist /= agent_dist.sum(dim=1, keepdim=True)
        ## similarity = (F.cosine_similarity(agent_dist, expert_dist).unsqueeze(1) + 1) / 2
        # similarity = (-((agent_dist * agent_dist.log()).sum(dim=1,keepdim=True) - torch.einsum('ik,jk->ij', agent_dist, expert_dist.log()))).exp().max(dim=1,keepdim=True)[0] # exp(-KLD) Eq(6)
        similarity = (-(agent_dist * (agent_dist.log() - expert_dist.log())).sum(dim=1, keepdim=True)).exp() # exp(-KLD) approximation Eq(7)
        if (type(self.sim_rate) == str) and ('auto' in self.sim_rate): # sim_rate should be like 'auto-1.0'
            self.sim_rate = float(self.sim_rate.split("-")[1]) / similarity.mean().item()
        similarity = self.sim_rate * similarity
        
        return similarity

    def update(self, replay_iter, expert_replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        self.global_step = step

        obs, action, reward_a, discount, next_obs = utils.to_torch(next(replay_iter), self.device)
        obs = obs.float()
        next_obs = next_obs.float()

        expert_obs, _, _, _, expert_next_obs = utils.to_torch(next(expert_replay_iter), self.device)
        expert_obs = expert_obs.float()
        expert_next_obs = expert_next_obs.float()

        obs_before_aug = obs
        next_obs_before_aug = next_obs
        expert_obs_before_aug = expert_obs
        expert_next_obs_before_aug = expert_next_obs

        # augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # disc augment
        disc_obs = self.disc_aug(obs_before_aug)
        disc_next_obs = self.disc_aug(next_obs_before_aug)
        disc_expert_obs = self.disc_aug(expert_obs_before_aug)
        disc_expert_next_obs = self.disc_aug(expert_next_obs_before_aug)
        
        results = self.update_discriminator(disc_obs, disc_expert_obs, disc_next_obs, disc_expert_next_obs)
        metrics.update(results)
        reward = self.dac_rewarder(disc_obs, disc_next_obs)

        similarity = self.compute_similarity(obs_before_aug, expert_obs_before_aug, next_obs_before_aug, expert_next_obs_before_aug)

        assert similarity.shape == reward.shape
        metrics['similarity'] = similarity.mean().item()

        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward_a.mean().item()

        if self.sim_type == "weight":
            new_rew = similarity * reward
        elif self.sim_type == "bonus":
            new_rew = similarity + reward
        else:
            raise NotImplementedError

        # update critic
        metrics.update(self.update_critic(obs, action, new_rew, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        
        metrics.update(self.record_grad_norm(self.critic, "critic"))
        metrics.update(self.record_grad_norm(self.discriminator, "discriminator"))
        metrics.update(self.record_grad_norm(self.encoder, "encoder"))

        return metrics

