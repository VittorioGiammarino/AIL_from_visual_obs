import threading

import dm_env
import gym
import functools
import numpy as np

from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

from dmc_remastered import ALL_ENVS
from dmc_remastered import DMCR_VARY

from DeepMind_control.dmc_expert import _spec_to_box, _flatten_obs

from utils_folder import utils_dreamer as utils

class DMC_Remastered_Env(dm_env.Environment):
    def __init__(self, 
                 task_builder,
                 visual_seed,
                 env_seed,
                 vary=DMCR_VARY):
        
        self._task_builder = task_builder
        self._env_seed = env_seed
        self._visual_seed = visual_seed
        
        self._env = self._task_builder(dynamics_seed=0, visual_seed=0, vary=vary)
        self._vary = vary
        
        self.make_new_env()
        
    def make_new_env(self):
        dynamics_seed = self._env_seed
        visual_seed = self._visual_seed
        self._env = self._task_builder(
            dynamics_seed=dynamics_seed, visual_seed=visual_seed, vary=self._vary,
        )
        
    def step(self, action):
        return self._env.step(action)
    
    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)    

class DeepMindControl:

  def __init__(self, name, action_repeat, seed, visual_seed, image_height=64, image_width=64, camera=None):
    domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)
    
    self._env = DMC_Remastered_Env(ALL_ENVS[domain][task], visual_seed, seed)
    self._action_repeat = action_repeat
    self._size = (image_height, image_width)
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera_id = camera

  def __getattr__(self, name):
      return getattr(self._env, name)   

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera_id)

class DeepMindControl_expert:

  def __init__(self, name, action_repeat, seed, visual_seed, image_height=64, image_width=64, camera=None):
    domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)
    
    self._env = DMC_Remastered_Env(ALL_ENVS[domain][task], visual_seed, seed)
    self._action_repeat = action_repeat
    self._size = (image_height, image_width)
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera_id = camera

    self._state_space = _spec_to_box(self._env.observation_spec().values(), np.float64)
    self.current_state = None

  def __getattr__(self, name):
      return getattr(self._env, name)  

  def _get_obs(self, time_step):
    obs = _flatten_obs(time_step.observation)
    return obs 

  @property
  def observation_space(self):
    spaces = {}
    spaces['state'] = self._state_space
    spaces['image'] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict()
    obs['state'] = self._get_obs(time_step)
    self.current_state = _flatten_obs(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict()
    obs['state'] = _flatten_obs(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera_id)

class CollectDataset:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    if isinstance(action, dict):
      transition.update(action)
    else:
      transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      for key, value in self._episode[1].items():
        if key not in self._episode[0]:
          self._episode[0][key] = 0 * value
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    # Missing keys will be filled with a zeroed out version of the first
    # transition, because we do not know what action information the agent will
    # pass yet.
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)

class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class SelectAction:
  def __init__(self, env, key):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    return self._env.step(action[self._key])

def process_episode(config, traindir, evaldir, mode, train_eps, eval_eps, episode):
  directory = dict(train=traindir, eval=evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = utils.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
  cache[str(filename)] = episode

def make_env(config, traindir, evaldir, mode, train_eps, eval_eps):
  env = DeepMindControl(config.task_name, config.action_repeat, config.seed, 
                      config.visual_seed, config.image_height, config.image_width)
  env = NormalizeActions(env)
  env = TimeLimit(env, config.time_limit)
  env = SelectAction(env, key='action')

  if (mode == 'train') or (mode == 'eval'):
    callbacks = [functools.partial(process_episode, config, traindir, evaldir, 
                                  mode, train_eps, eval_eps)]
    env = CollectDataset(env, callbacks)

  env = RewardObs(env)

  return env

def process_episode_expert(expertdir, expert_eps, episode):
  directory = expertdir
  cache = expert_eps
  filename = utils.save_episodes(directory, [episode])[0]
  cache.clear()
  cache[str(filename)] = episode

def make_env_expert(config, expertdir, expert_eps):
  env = DeepMindControl_expert(config.task_name, config.action_repeat_source, config.seed, 
                              config.visual_seed_source, config.image_height, config.image_width)

  env = NormalizeActions(env)
  env = TimeLimit(env, config.time_limit)

  callbacks = [functools.partial(process_episode_expert, expertdir, expert_eps)]

  env = CollectDataset(env, callbacks)
  env = RewardObs(env)

  return env
