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

def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

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
                 frame_skip=1,
                 height=84,
                 width=84,
                 camera_id=0,
                 num_frames=3,
                 max_episode_steps=1000):
        
        self._env_seed = env_seed
        self._max_episode_steps = max_episode_steps
        
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
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
        obs = _flatten_obs(time_step.observation)
        return obs

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
    def visual_observation_space(self):
        return (3*self._num_frames, self._width, self._height)  

    @property
    def reward_range(self):
        return 0, self._frame_skip

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

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break

        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra, time_step

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        
        pixels = self._extract_pixels()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        
        obs = self._get_obs(time_step)
        return obs, time_step

    def render(self, mode='rgb_array', height=None, width=None, camera_id=None):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
    
    def step_learn_from_pixels(self, time_step, action=None):
        pixels = self._extract_pixels()
        self._frames.append(pixels)
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        new_time_step = time_step._replace(observation=obs)
        if action is None:
            action_spec = self._env.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=new_time_step.observation,
                                step_type=new_time_step.step_type,
                                action=action,
                                reward=new_time_step.reward or 0.0,
                                discount=new_time_step.discount or 1.0)

def make_states_only(domain_name,
                    task_name, 
                    seed, 
                    frame_skip=1,
                    height=84,
                    width=84,
                    camera_id=0,
                    num_frames=3,
                    max_episode_steps=1000):

    if domain_name == 'quadruped':
        camera_id = 2
    elif domain_name == 'dog':
        camera_id = 0
            
    env = DMC_Wrapper(domain_name,
                    task_name, 
                    seed,
                    height = height,
                    width = width,
                    camera_id = camera_id,
                    frame_skip = frame_skip,
                    num_frames = num_frames,
                    max_episode_steps = max_episode_steps)

    return env

