# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from collections import deque

import hydra
import numpy as np
import torch
from dm_env import specs

from DeepMind_control import dmc, dmc_expert
from utils_folder import utils
from logger_folder.logger import Logger
from video import TrainVideoRecorder, VideoRecorder
from buffers.replay_buffer_latent_model import ReplayBuffer as ReplayBufferAgent
from buffers.replay_buffer_latent_model_expert import ReplayBuffer as ReplayBufferExpert 

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.state_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

def make_env_expert(cfg):
    """Helper function to create dm_control environment"""
    domain, task = cfg.task_name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)

    env = dmc_expert.make_states_only(domain_name=domain,
                                    task_name=task,
                                    seed=cfg.seed,
                                    frame_skip=cfg.frame_skip,
                                    height=cfg.image_height,
                                    width=cfg.image_width,
                                    camera_id=0,
                                    num_frames = cfg.frame_stack)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env

class LMObservation:
    """
    Observation Latent model.
    """
    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, time_step):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(time_step.observation)

    def append(self, time_step):
        self._state.append(time_step.observation)
        self._action.append(time_step.action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create target envs and agent
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed,
                                self.cfg.image_height, self.cfg.image_width)

        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed,
                                self.cfg.image_height, self.cfg.image_width)

        self.replay_buffer = ReplayBufferAgent(self.cfg.replay_buffer_size, self.cfg.sequence_length, 
                                            self.train_env.observation_spec().shape, self.train_env.action_spec().shape, 
                                            self.cfg.device)

        #create source envs and agent
        self.expert_env = make_env_expert(self.cfg)
        self.cfg.expert.obs_dim = self.expert_env.observation_space.shape[0]
        self.cfg.expert.action_dim = self.expert_env.action_space.shape[0]
        self.cfg.expert.action_range = [float(self.expert_env.action_space.low.min()),
                                        float(self.expert_env.action_space.high.max())]
        
        self.expert = hydra.utils.instantiate(self.cfg.expert)
        
        self.replay_buffer_expert = ReplayBufferExpert(self.cfg.replay_buffer_size, self.cfg.sequence_length, 
                                                    self.train_env.observation_spec().shape, self.train_env.action_spec().shape, 
                                                    self.cfg.device)

        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if self.cfg.save_train_video else None)

        self.ob = LMObservation(self.train_env.observation_spec().shape, self.train_env.action_spec().shape, self.cfg.sequence_length)
        self.ob_test = LMObservation(self.train_env.observation_spec().shape, self.train_env.action_spec().shape, self.cfg.sequence_length)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    def store_expert_transitions(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
        
        while eval_until_episode(episode):
            obs, time_step = self.expert_env.reset()
            self.expert.reset()
            self.video_recorder.init(self.expert_env, enabled=(episode == 0))
            
            extended_time_step = self.expert_env.step_learn_from_pixels(time_step)
            self.replay_buffer_expert.reset_episode(extended_time_step)

            done = False
            
            while not done:
                with torch.no_grad(), utils.eval_mode(self.expert):
                    action = self.expert.act(obs, self.global_step, eval_mode=True)
                obs, reward, done, _, time_step = self.expert_env.step(action)    
                
                extended_time_step = self.expert_env.step_learn_from_pixels(time_step, action)
                self.replay_buffer_expert.append(extended_time_step, done)
                self.video_recorder.record(self.expert_env)
                
                total_reward += extended_time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save('expert.mp4')

        print(f'Average expert reward: {total_reward / episode}, Total number of samples: {step}')
                
    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.ob_test.reset_episode(time_step)
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(self.ob_test, self.global_step, eval_mode=True)
                
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                self.ob_test.append(time_step)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        train_latent_only_until_step = utils.Until(self.cfg.num_seed_frames_update_latent_only,
                                                    self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.reset_episode(time_step)
        self.ob.reset_episode(time_step)

        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_buffer.reset_episode(time_step)
                self.ob.reset_episode(time_step)

                self.train_video_recorder.init(time_step.observation)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(self.ob, self.global_step, eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):

                if not train_latent_only_until_step(self.global_step):
                    metrics = self.agent.update(self.replay_buffer, self.replay_buffer_expert, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

                else:
                    _ = self.agent.update_latent(self.replay_buffer)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.ob.append(time_step)
            self.replay_buffer.append(time_step, time_step.last())
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        
    def load_expert(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.expert = payload['agent']

@hydra.main(config_path='config_folder/POMDP', config_name='config_lail')
def main(cfg):
    from train_LAIL import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    parent_dir = root_dir.parents[3]
    snapshot = parent_dir / f'expert_policies/snapshot_{cfg.task_name}_frame_skip_{cfg.frame_skip}.pt'
    assert snapshot.exists()
    print(f'loading expert target: {snapshot}')
    workspace.load_expert(snapshot)
    workspace.store_expert_transitions()
    workspace.train()

if __name__ == '__main__':
    main()