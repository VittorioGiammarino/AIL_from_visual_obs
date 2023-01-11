import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
from torch import autograd
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

from utils_folder import utils_dreamer as utils
import agents.exploration_dreamer as expl

to_np = lambda x: x.detach().cpu().numpy()

class RSSM(nn.Module):

  def __init__(self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1, rec_depth=1, shared=False, discrete=False, act=nn.ELU,
              mean_act='none', std_act='softplus', temp_post=True, min_std=0.1, cell='gru', num_actions=None, embed = None, device=None):

    super(RSSM, self).__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = embed
    self._device = device

    inp_layers = []

    if self._discrete:
      inp_dim = self._stoch * self._discrete + num_actions
    else:
      inp_dim = self._stoch + num_actions

    if self._shared:
      inp_dim += self._embed

    for i in range(self._layers_input):
      inp_layers.append(nn.Linear(inp_dim, self._hidden))
      inp_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden

    self._inp_layers = nn.Sequential(*inp_layers)

    if cell == 'gru':
      self._cell = GRUCell(self._hidden, self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._hidden, self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

    img_out_layers = []
    inp_dim = self._deter
    for i in range(self._layers_output):
      img_out_layers.append(nn.Linear(inp_dim, self._hidden))
      img_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden

    self._img_out_layers = nn.Sequential(*img_out_layers)

    obs_out_layers = []
    if self._temp_post:
      inp_dim = self._deter + self._embed
    else:
      inp_dim = self._embed

    for i in range(self._layers_output):
      obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
      obs_out_layers.append(self._act())
      if i == 0:
        inp_dim = self._hidden

    self._obs_out_layers = nn.Sequential(*obs_out_layers)

    if self._discrete:
      self._ims_stat_layer = nn.Linear(self._hidden, self._discrete*self._stoch)
      self._obs_stat_layer = nn.Linear(self._hidden, self._discrete*self._stoch)
    else:
      self._ims_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
      self._obs_stat_layer = nn.Linear(self._hidden, 2*self._stoch)
 
  def initial(self, batch_size):
    deter = torch.zeros(batch_size, self._deter).to(self._device)
    if self._discrete:
      state = dict(logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                  stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                  deter=deter)
    else:
      state = dict(mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                  std=torch.zeros([batch_size, self._stoch]).to(self._device),
                  stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                  deter=deter)
    return state

  def observe(self, embed, action, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))

    if state is None:
      state = self.initial(action.shape[0])

    embed, action = swap(embed), swap(action)
    post, prior = utils.static_scan(lambda prev_state, prev_act, embed: self.obs_step(prev_state[0], prev_act, embed), (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))

    if state is None:
      state = self.initial(action.shape[0])

    assert isinstance(state, dict), state
    action = action
    action = swap(action)
    prior = utils.static_scan(self.img_step, [action], state)
    prior = prior[0]
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      dist = torchd.independent.Independent(utils.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = utils.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
    return dist

  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action, None, sample)

    if self._shared:
      post = self.img_step(prev_state, prev_action, embed, sample)
    else:
      if self._temp_post:
        x = torch.cat([prior['deter'], embed], -1)
      else:
        x = embed
      x = self._obs_out_layers(x)

      stats = self._suff_stats_layer('obs', x)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()

      post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def img_step(self, prev_state, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']

    if self._discrete:
      shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
      prev_stoch = prev_stoch.reshape(shape)

    if self._shared:
      if embed is None:
        shape = list(prev_action.shape[:-1]) + [self._embed]
        embed = torch.zeros(shape)
      x = torch.cat([prev_stoch, prev_action, embed], -1)
    else:
      x = torch.cat([prev_stoch, prev_action], -1)
    x = self._inp_layers(x)

    for _ in range(self._rec_depth): # rec depth is not correctly implemented
      deter = prev_state['deter']
      x, deter = self._cell(x, [deter])
      deter = deter[0]  # Keras wraps the state in a list.
    x = self._img_out_layers(x)
    stats = self._suff_stats_layer('ims', x)

    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):

    if self._discrete:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}

    else:
      if name == 'ims':
        x = self._ims_stat_layer(x)
      elif name == 'obs':
        x = self._obs_stat_layer(x)
      else:
        raise NotImplementedError

      mean, std = torch.split(x, [self._stoch]*2, -1)
      mean = {'none': lambda: mean, 'tanh5': lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
      std = {'softplus': lambda: torch.softplus(std), 'abs': lambda: torch.abs(std + 1), 'sigmoid': lambda: torch.sigmoid(std), 
            'sigmoid2': lambda: 2 * torch.sigmoid(std / 2)}[self._std_act]()

      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, scale):
    kld = torchd.kl.kl_divergence
    dist = lambda x: self.get_dist(x)
    sg = lambda x: {k: v.detach() for k, v in x.items()} # stop gradient
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                  dist(rhs) if self._discrete else dist(rhs)._dist)
      loss = torch.mean(torch.maximum(value, free))
    else:
      value_lhs = value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
                              dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist)
      value_rhs = kld(dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                      dist(rhs) if self._discrete else dist(rhs)._dist)
      loss_lhs = torch.maximum(torch.mean(value_lhs), torch.Tensor([free])[0])
      loss_rhs = torch.maximum(torch.mean(value_rhs), torch.Tensor([free])[0])
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    loss *= scale
    return loss, value


class ConvEncoder(nn.Module):

  def __init__(self, grayscale=False, depth=32, act=nn.ReLU, kernels=(4, 4, 4, 4)):
    super(ConvEncoder, self).__init__()
    self._act = act
    self._depth = depth
    self._kernels = kernels

    layers = []
    for i, kernel in enumerate(self._kernels):
      if i == 0:
        if grayscale:
          inp_dim = 1
        else:
          inp_dim = 3
      else:
        inp_dim = 2 ** (i-1) * self._depth
      depth = 2 ** i * self._depth
      layers.append(nn.Conv2d(inp_dim, depth, kernel, 2))
      layers.append(act())
    self.layers = nn.Sequential(*layers)

  def __call__(self, obs):
    x = obs['image'].reshape((-1,) + tuple(obs['image'].shape[-3:]))
    x = x.permute(0, 3, 1, 2)
    x = self.layers(x)
    x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
    shape = list(obs['image'].shape[:-3]) + [x.shape[-1]]
    return x.reshape(shape)


class ConvDecoder(nn.Module):

  def __init__(self, inp_depth, depth=32, act=nn.ReLU, shape=(3, 64, 64), kernels=(5, 5, 6, 6), thin=True):
    super(ConvDecoder, self).__init__()
    self._inp_depth = inp_depth
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

    if self._thin:
      self._linear_layer = nn.Linear(inp_depth, 32 * self._depth)
    else:
      self._linear_layer = nn.Linear(inp_depth, 128 * self._depth)
    inp_dim = 32 * self._depth

    cnnt_layers = []
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        #depth = self._shape[-1]
        depth = self._shape[0]
        act = None
      if i != 0:
        inp_dim = 2 ** (len(self._kernels) - (i-1) - 2) * self._depth
      cnnt_layers.append(nn.ConvTranspose2d(inp_dim, depth, kernel, 2))
      if act is not None:
        cnnt_layers.append(act())
    self._cnnt_layers = nn.Sequential(*cnnt_layers)

  def __call__(self, features, dtype=None):
    if self._thin:
      x = self._linear_layer(features)
      x = x.reshape([-1, 1, 1, 32 * self._depth])
      x = x.permute(0,3,1,2)
    else:
      x = self._linear_layer(features)
      x = x.reshape([-1, 2, 2, 32 * self._depth])
      x = x.permute(0,3,1,2)
    x = self._cnnt_layers(x)
    mean = x.reshape(features.shape[:-1] + self._shape)
    mean = mean.permute(0, 1, 3, 4, 2)
    return utils.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, 1), len(self._shape)))


class DenseHead(nn.Module):

  def __init__(self, inp_dim, shape, layers, units, act=nn.ELU, dist='normal', std=1.0):
    super(DenseHead, self).__init__()
    self._shape = (shape,) if isinstance(shape, int) else shape
    if len(self._shape) == 0:
      self._shape = (1,)
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

    if self._std == 'learned':
      self._std_layer = nn.Linear(inp_dim, np.prod(self._shape)) #this needs to be double checked with the original code

    mean_layers = []
    for index in range(self._layers):
      mean_layers.append(nn.Linear(inp_dim, self._units))
      mean_layers.append(act())
      if index == 0:
        inp_dim = self._units
    mean_layers.append(nn.Linear(inp_dim, np.prod(self._shape)))
    self._mean_layers = nn.Sequential(*mean_layers)

  def __call__(self, features, dtype=None):
    x = features
    mean = self._mean_layers(x)
    if self._std == 'learned':
      std = self._std_layer(x)
      std = F.softplus(std) + 0.01
    else:
      std = self._std
    if self._dist == 'normal':
      return utils.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), len(self._shape)))
    if self._dist == 'huber':
      return utils.ContDist(torchd.independent.Independent(utils.UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
    if self._dist == 'binary':
      return utils.Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
    raise NotImplementedError(self._dist)


class ActionHead(nn.Module):

  def __init__(self, inp_dim, size, layers, units, act=nn.ELU, dist='trunc_normal', init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    super(ActionHead, self).__init__()
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

    pre_layers = []
    for index in range(self._layers):
      pre_layers.append(nn.Linear(inp_dim, self._units))
      pre_layers.append(act())
      if index == 0:
        inp_dim = self._units
    self._pre_layers = nn.Sequential(*pre_layers)

    if self._dist in ['tanh_normal','tanh_normal_5','normal','trunc_normal']:
      self._dist_layer = nn.Linear(self._units, 2 * self._size)
    elif self._dist in ['normal_1','onehot','onehot_gumbel']:
      self._dist_layer = nn.Linear(self._units, self._size)

  def __call__(self, features, dtype=None):
    x = features
    x = self._pre_layers(x)
    if self._dist == 'tanh_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = torch.tanh(mean)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(dist, utils.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = utils.SampleDist(dist)

    elif self._dist == 'tanh_normal_5':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      mean = 5 * torch.tanh(mean / 5)
      std = F.softplus(std + 5) + 5
      dist = torchd.normal.Normal(mean, std)
      dist = torchd.transformed_distribution.TransformedDistribution(dist, utils.TanhBijector())
      dist = torchd.independent.Independent(dist, 1)
      dist = utils.SampleDist(dist)

    elif self._dist == 'normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, 2, -1)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = torchd.normal.Normal(mean, std)
      dist = utils.ContDist(torchd.independent.Independent(dist, 1))

    elif self._dist == 'normal_1':
      x = self._dist_layer(x)
      dist = torchd.normal.Normal(mean, 1)
      dist = utils.ContDist(torchd.independent.Independent(dist, 1))

    elif self._dist == 'trunc_normal':
      x = self._dist_layer(x)
      mean, std = torch.split(x, [self._size]*2, -1)
      mean = torch.tanh(mean)
      std = 2 * torch.sigmoid(std / 2) + self._min_std
      dist = utils.SafeTruncatedNormal(mean, std, -1, 1)
      dist = utils.ContDist(torchd.independent.Independent(dist, 1))
    else:
      raise NotImplementedError(self._dist)
    return dist


class GRUCell(nn.Module):

  def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
    super(GRUCell, self).__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(inp_size+size, 3*size, bias=norm is not None)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = torch.split(parts, [self._size]*3, -1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class WorldModel(nn.Module):

  def __init__(self, step, action_shape, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config
    act = getattr(torch.nn, config.act)
    self.action_shape = action_shape

    self.encoder = ConvEncoder(config.grayscale, config.cnn_depth, act, config.encoder_kernels)

    if config.image_height == 64 and config.image_width == 64:
      embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      embed_size *= 2 * 2
    else:
      raise NotImplemented(f"{config.size} is not applicable now")

    self.dynamics = RSSM(config.dyn_stoch, config.dyn_deter, config.dyn_hidden, config.dyn_input_layers, config.dyn_output_layers,
                        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete, act, config.dyn_mean_act, config.dyn_std_act,
                        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell, self.action_shape, embed_size, config.device)

    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    shape = (channels,) + (config.image_height, config.image_width)

    if config.dyn_discrete:
      feat_size = config.dyn_stoch*config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter

    self.heads['image'] = ConvDecoder(feat_size, config.cnn_depth, act, shape, config.decoder_kernels, config.decoder_thin)
    self.heads['reward'] = DenseHead(feat_size, [], config.reward_layers, config.units, act)
    if config.pred_discount:
      self.heads['discount'] = DenseHead(feat_size, [], config.discount_layers, config.units, act, dist='binary')
    for name in config.grad_heads:
      assert name in self.heads, name
    self._model_opt = utils.Optimizer('model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
                                      config.weight_decay, opt=config.opt,use_amp=self._use_amp)

    self._scales = dict(reward=config.reward_scale, discount=config.discount_scale)

  def _train(self, data):
    data = self.preprocess(data)

    with utils.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data['action'])
        kl_balance = utils.schedule(self._config.kl_balance, self._step)
        kl_free = utils.schedule(self._config.kl_free, self._step)
        kl_scale = utils.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        losses = {}
        likes = {}
        for name, head in self.heads.items():
          grad_head = (name in self._config.grad_heads)
          feat = self.dynamics.get_feat(post)
          feat = feat if grad_head else feat.detach()
          pred = head(feat)
          like = pred.log_prob(data[name])
          likes[name] = like
          losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
        model_loss = sum(losses.values()) + kl_loss
      metrics = self._model_opt(model_loss, self.parameters())

    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = to_np(torch.mean(kl_value))
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
      context = dict(embed=embed, feat=self.dynamics.get_feat(post), kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics

  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image'].copy()) / 255.0 - 0.5

    if self._config.clip_rewards == 'tanh':
      obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
    elif self._config.clip_rewards == 'identity':
      obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_rewards} is not implemented')

    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs

  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data)

    states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    recon = self.heads['image'](self.dynamics.get_feat(states)).mode()[:6]
    reward_post = self.heads['reward'](self.dynamics.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics.imagine(data['action'][:6, 5:], init)
    openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
    reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):

  def __init__(self, config, action_shape, world_model, actor_entropy, actor_state_entropy, imag_gradient_mix, stop_grad_actor=True, reward=None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    act = getattr(torch.nn, config.act)
    self.action_shape = action_shape

    self.actor_entropy = actor_entropy
    self.actor_state_entropy = actor_state_entropy
    self.imag_gradient_mix = imag_gradient_mix
    self.GAN_loss = config.GAN_loss

    if config.dyn_discrete:
      feat_size = config.dyn_stoch*config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter

    if config.image_height == 64 and config.image_width == 64:
      embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      embed_size *= 2 * 2
    else:
      raise NotImplemented(f"{config.size} is not applicable now")

    self.actor = ActionHead(feat_size, self.action_shape, config.actor_layers, config.units, act,
                            config.actor_dist, config.actor_init_std, config.actor_min_std, config.actor_dist, config.actor_temp, config.actor_outscale)

    self.value = DenseHead(feat_size, [], config.value_layers, config.units, act, config.value_head)

    if config.GAN_loss == 'bce':
      self.discriminator = DenseHead(2*embed_size, [], config.discriminator_layers, config.units, act, dist='binary')
    elif config.GAN_loss == 'least-square':
      self.discriminator = DenseHead(2*embed_size, [], config.discriminator_layers, config.units, act, dist='normal', std='learned')
      self.reward_d_coef = config.reward_d_coef
    else:
      NotImplementedError

    if config.slow_value_target or config.slow_actor_target:
      self._slow_value = DenseHead(feat_size, [], config.value_layers, config.units, act)
      self._updates = 0

    kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)

    self._actor_opt = utils.Optimizer('actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip, **kw)
    self._value_opt = utils.Optimizer('value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,**kw)
    self._discriminator_opt = utils.Optimizer('discriminator', self.discriminator.parameters(), config.discriminator_lr, 
                                              config.opt_eps, config.discriminator_grad_clip,**kw)

  def _train(self, start, agent_data, expert_data, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None):
    objective = objective or self._reward
    self._update_slow_target()
    metrics = {}

    expert_data = self._world_model.preprocess(expert_data)
    agent_data = self._world_model.preprocess(agent_data)

    with utils.RequiresGrad(self.actor), utils.RequiresGrad(self.discriminator):
      with torch.cuda.amp.autocast(self._use_amp):

        embed_expert = self._world_model.encoder(expert_data)
        embed_agent = self._world_model.encoder(agent_data)

        feat_expert_dist = torch.cat([embed_expert[:, :-1, :], embed_expert[:, 1:, :]], -1)
        feat_policy_dist = torch.cat([embed_agent[:, :-1, :], embed_agent[:, 1:, :]], -1)

        expert_d = self.discriminator(feat_expert_dist)
        policy_d = self.discriminator(feat_policy_dist.detach())

        if self.GAN_loss == 'bce':
          expert_loss = (expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self._config.device))).mean()
          policy_loss = (policy_d.log_prob(torch.zeros_like(policy_d.mode()).to(self._config.device))).mean()
          grad_penalty = self.compute_discriminator_grad_penalty(feat_policy_dist.detach(), feat_expert_dist.detach())
          discriminator_loss = -(expert_loss+policy_loss) + grad_penalty
          reward = policy_d.mode().detach()

        elif self.GAN_loss == 'least-square':
          expert_labels = 1.0
          agent_labels = -1.0

          expert_loss = F.mse_loss(expert_d.sample(), expert_labels*torch.ones_like(expert_d.mode()).to(self._config.device))
          policy_loss = F.mse_loss(policy_d.sample(), agent_labels*torch.zeros_like(policy_d.mode()).to(self._config.device))
          grad_penalty = self.compute_discriminator_grad_penalty_LS(feat_expert_dist.detach())
          discriminator_loss = 0.5*(expert_loss+policy_loss) + grad_penalty
          reward = self.compute_reward_LS(feat_policy_dist.detach())
        
        state = {k: v[:,:-1,:] for k, v in start.items()}
        feat = self._world_model.dynamics.get_feat(state) 
        action = self.actor(feat.detach() if self._stop_grad_actor else feat)

        actor_ent = self.actor(feat).entropy()
        state_ent = self._world_model.dynamics.get_dist(state).entropy()
        target, weights = self._compute_target(feat, state, action, reward, actor_ent, state_ent, self._config.slow_actor_target)
        actor_loss, mets = self._compute_actor_loss(feat, state, action, target, actor_ent, state_ent, weights)
        metrics.update(mets)

        if self._config.slow_value_target != self._config.slow_actor_target:
          target, weights = self._compute_target(feat, state, action, reward, actor_ent, state_ent, self._config.slow_value_target)
        value_input = feat

    with utils.RequiresGrad(self.value):
      with torch.cuda.amp.autocast(self._use_amp):
        value = self.value(value_input[:-1].detach())
        target = torch.stack(target, dim=1)
        value_loss = -value.log_prob(target.detach())
        if self._config.value_decay:
          value_loss += self._config.value_decay * value.mode()
        value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])

    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))
    metrics['actor_ent'] = to_np(torch.mean(actor_ent))

    with utils.RequiresGrad(self):
      metrics.update(self._discriminator_opt(discriminator_loss, self.discriminator.parameters()))
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      metrics.update(self._value_opt(value_loss, self.value.parameters()))

    return feat, state, action, weights, metrics

  def _imagine(self, start, policy, horizon, repeats=None):
    dynamics = self._world_model.dynamics

    if repeats:
      raise NotImplemented("repeats is not implemented in this version")

    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}

    def step(prev, _):
      state, _, _ = prev
      feat = dynamics.get_feat(state)
      inp = feat.detach() if self._stop_grad_actor else feat
      action = policy(inp).sample()
      succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
      return succ, feat, action

    feat = 0 * dynamics.get_feat(start)
    action = policy(feat).mode()
    succ, feats, actions = utils.static_scan(step, [torch.arange(horizon)], (start, feat, action))
    states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

    if repeats:
      raise NotImplemented("repeats is not implemented in this version")

    return feats, states, actions

  def _compute_target(self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent, slow):

    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(reward)

    if self._config.future_entropy and self.actor_entropy() > 0:
      reward += self.actor_entropy() * actor_ent

    if self._config.future_entropy and self.actor_state_entropy() > 0:
      reward += self.actor_state_entropy() * state_ent

    if slow:
      value = self._slow_value(imag_feat).mode()
    else:
      value = self.value(imag_feat).mode()

    target = utils.lambda_return(reward[:-1], value[:-1], discount[:-1], bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()

    return target, weights

  def _compute_actor_loss(self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent, weights):
    metrics = {}
    inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
    policy = self.actor(inp)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)

    if self._config.imag_gradient == 'dynamics':
      actor_target = target
    elif self._config.imag_gradient == 'reinforce':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
    elif self._config.imag_gradient == 'both':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
      mix = self.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix
    else:
      raise NotImplementedError(self._config.imag_gradient)

    if not self._config.future_entropy and (self.actor_entropy() > 0):
      actor_target += self.actor_entropy() * actor_ent[:-1][:,:,None]

    if not self._config.future_entropy and (self.actor_state_entropy() > 0):
      actor_target += self.actor_state_entropy() * state_ent[:-1]

    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1

  def compute_discriminator_grad_penalty(self, policy_feat, expert_feat, lambda_=10):
      alpha = torch.rand(policy_feat.shape[:2]).unsqueeze(-1).to(self._config.device)
      disc_penalty_input = alpha*policy_feat + (1-alpha)*expert_feat
      disc_penalty_input.requires_grad = True
      d = self.discriminator(disc_penalty_input).mode()
      ones = torch.ones(d.size(), device=self._config.device)
      grad = autograd.grad(outputs=d, inputs=disc_penalty_input, grad_outputs=ones, create_graph=True,
                          retain_graph=True, only_inputs=True)[0]
      
      grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
      return grad_pen

  def compute_discriminator_grad_penalty_LS(self, expert_feat, lambda_=10):
      expert_feat.requires_grad = True
      d = self.discriminator(expert_feat).mode()
      ones = torch.ones(d.size(), device=self._config.device)
      grad = autograd.grad(outputs=d, inputs=expert_feat, grad_outputs=ones, create_graph=True,
                          retain_graph=True, only_inputs=True)[0]
      
      grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
      return grad_pen

  def compute_reward_LS(self, policy_feat, reward_a=0):
    with torch.no_grad():
      self.discriminator.eval()
      d = self.discriminator(policy_feat).sample()
      reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
      self.discriminator.train()
    return reward_d


def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))

class VmaifoAgent(nn.Module):
  def __init__(self, action_shape, device, dataset, traindir, expert_dataset, expertdir, config):
    super(VmaifoAgent, self).__init__()

    self.action_shape = action_shape[0]
    self.device = device
    self._config = config
    self._step = count_steps(traindir)

    # Schedules.
    actor_entropy = (lambda x=config.actor_entropy: utils.schedule(x, self._step))
    actor_state_entropy = (lambda x=config.actor_state_entropy: utils.schedule(x, self._step))
    imag_gradient_mix = (lambda x=config.imag_gradient_mix: utils.schedule(x, self._step))

    self.dataset = dataset
    self.expert_dataset = expert_dataset
    self.expert_steps = count_steps(expertdir)

    self.world_model = WorldModel(self._step, self.action_shape, config)
    self.task_behavior = ImagBehavior(config, self.action_shape, self.world_model, actor_entropy, 
                                      actor_state_entropy, imag_gradient_mix, config.behavior_stop_grad)

    reward = lambda f, s, a: self.world_model.heads['reward'](f).mean

    self._expl_behavior = dict(greedy=lambda: self.task_behavior, 
                              random=lambda: expl.Random(config),)[config.expl_behavior]()

  def train(self, training=True):
    self.training=training
    self.world_model.train(training)
    self.task_behavior.train(training)

  def act(self, obs, agent_state, step, eval_mode):
    if agent_state is None:
      latent = self.world_model.dynamics.initial(1)
      action = torch.zeros((1, self.action_shape)).to(self.device)
    else:
      latent, action = agent_state

    obs['image'] = (torch.Tensor(obs['image'].copy()) / 255.0 - 0.5).unsqueeze(0).to(self.device)

    embed = self.world_model.encoder(obs)
    latent, _ = self.world_model.dynamics.obs_step(latent, action, embed, self._config.collect_dyn_sample)
    feat = self.world_model.dynamics.get_feat(latent)
    actor = self.task_behavior.actor(feat)

    if eval_mode:
      action = actor.mode()
    else:
      actor = self.task_behavior.actor(feat)
      action = actor.sample()
      if step < self._config.num_expl_steps:
        action.uniform_(-1.0, 1.0)

    logprob = actor.log_prob(action)
    latent = {k: v.detach()  for k, v in latent.items()}
    action = action.detach()
    state = (latent, action)
    policy_output = {'action': action.cpu().numpy()[0], 'logprob': logprob.cpu().numpy()[0]}

    return policy_output, state

  def pretrain(self, steps):
    metrics = dict()

    for _ in range(steps):
      data = next(self.dataset)

      post, context, mets = self.world_model._train(data)

      start = post

      if self._config.pred_discount:  # Last step could be terminal.
        start = {k: v[:, :-1] for k, v in post.items()}
        context = {k: v[:, :-1] for k, v in context.items()}

      reward = lambda f, s, a: self.world_model.heads['reward'](self.world_model.dynamics.get_feat(s)).mode()
      expert_data = next(self.expert_dataset)
      mets_behavior = self.task_behavior._train(start, data, expert_data, reward)[-1]

    metrics.update(mets)
    metrics.update(mets_behavior)

    return metrics

  def update(self, step):
    metrics = dict()

    if step % self._config.train_every != 0:
      return metrics

    data = next(self.dataset)

    post, context, mets = self.world_model._train(data)
    metrics.update(mets)

    start = post

    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}

    reward = lambda f, s, a: self.world_model.heads['reward'](self.world_model.dynamics.get_feat(s)).mode()
    expert_data = next(self.expert_dataset)
    metrics.update(self.task_behavior._train(start, data, expert_data, reward)[-1])

    return metrics



