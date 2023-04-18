# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from typing import Any, NamedTuple

from gym import core, spaces
from dm_env import StepType, specs
import numpy as np

from dm_control import suite

def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class DMC_Wrapper(core.Env):
    def __init__(self, 
                 domain_name,
                 task_name,
                 env_seed,
                 observables = ["velocity"],
                 actions_observable = False,
                 state_skip=1,
                 height=84,
                 width=84,
                 camera_id=0,
                 num_states=3,
                 max_episode_steps=1000):
        
        self._env_seed = env_seed
        self._max_episode_steps = max_episode_steps

        self.observables = observables
        self.action_observable = actions_observable
        
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._state_skip = state_skip

        if actions_observable:
            self._num_states = 2*num_states
        else:
            self._num_states = num_states

        self._states = deque([], maxlen=self._num_states)
        
        self._env = suite.load(domain_name,
                            task_name,
                            task_kwargs={'random': env_seed},
                            visualize_reward=False)
        
        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        self._observation_space = _spec_to_box(self._env.observation_spec().values(), np.float64)
        self._state_space = _spec_to_box(self._env.observation_spec().values(), np.float64)
        
        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = self._flatten_obs(time_step.observation)
        return obs

    def _flatten_obs(self, obs):
        obs_pieces = []
        for k,v in obs.items():

            if k in self.observables:
                flat = np.array([v]) if np.isscalar(v) else v.ravel()
                obs_pieces.append(flat)
            else:
                flat = np.array([0]) if np.isscalar(v) else np.zeros_like(v)
                obs_pieces.append(flat)

        return np.concatenate(obs_pieces, axis=0)

    def _get_obs_full(self, time_step):
        obs = self._flatten_obs_full(time_step.observation)
        return obs

    def _flatten_obs_full(self, obs):
        obs_pieces = []
        print(obs)
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0)

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action
    
    def _extract_pixels(self):
        obs = self.render(height=self._height, width=self._width, camera_id=self._camera_id)
        obs = obs.transpose(2, 0, 1).copy()
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._state_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._state_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break

        obs_partial = self._get_obs(time_step)

        if self.action_observable:
            self._states.append(obs_partial)
            self._states.append(action)
        else:
            self._states.append(obs_partial)

        obs_partial = np.concatenate(list(self._states), axis=0)

        self.current_state = self._flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs_partial, reward, done, extra, time_step

    def reset(self):
        time_step = self._env.reset()
        self.current_state = self._flatten_obs(time_step.observation)
        
        obs_partial = self._get_obs(time_step)

        if self.action_observable:
            for _ in range(self._num_states):
                self._states.append(obs_partial)
                self._states.append(np.zeros(self.action_space.shape))
        else:
            for _ in range(self._num_states):
                self._states.append(obs_partial)

        obs_partial = np.concatenate(list(self._states), axis=0)
        
        return obs_partial, time_step

    def step_full(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._state_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break

        obs_partial = self._get_obs(time_step)

        if self.action_observable:
            self._states.append(obs_partial)
            self._states.append(action)
        else:
            self._states.append(obs_partial)

        obs_partial = np.concatenate(list(self._states), axis=0)

        obs = self._get_obs_full(time_step)

        self.current_state = self._flatten_obs_full(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, obs_partial, reward, done, extra, time_step

    def reset_full(self):
        time_step = self._env.reset()
        self.current_state = self._flatten_obs_full(time_step.observation)

        obs_partial = self._get_obs(time_step)

        if self.action_observable:
            for _ in range(self._num_states):
                self._states.append(obs_partial)
                self._states.append(np.zeros(self.action_space.shape))
        else:
            for _ in range(self._num_states):
                self._states.append(obs_partial)

        obs_partial = np.concatenate(list(self._states), axis=0)

        obs = self._get_obs_full(time_step)

        return obs, obs_partial, time_step

    def render(self, mode='rgb_array', height=None, width=None, camera_id=None):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

def make_states_only(domain_name,
                    task_name, 
                    seed, 
                    observables = ['velocity'],
                    action_observable = False,
                    state_skip=1,
                    height=84,
                    width=84,
                    camera_id=0,
                    num_states=3,
                    max_episode_steps=1000):

    if domain_name == 'quadruped':
        camera_id = 2
    elif domain_name == 'dog':
        camera_id = 0
            
    env = DMC_Wrapper(domain_name,
                    task_name, 
                    seed,
                    observables = observables,
                    actions_observable = action_observable,
                    state_skip = state_skip,
                    height = height,
                    width = width,
                    camera_id = camera_id,
                    num_states = num_states,
                    max_episode_steps = max_episode_steps)

    return env

