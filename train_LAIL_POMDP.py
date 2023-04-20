#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import time
import torch
import numpy as np

from pathlib import Path
from collections import deque

from video import VideoRecorder_Expert as VideoRecorder
from logger_folder.logger import Logger
from utils_folder import utils 

from buffers.replay_buffer_latent_model_no_visual import ReplayBuffer as ReplayBufferAgent
from buffers.replay_buffer_latent_model_expert_no_visual import ReplayBuffer as ReplayBufferExpert 

from DeepMind_control import dmc_POMDP
import hydra

torch.autograd.set_detect_anomaly(False)

def make_env(cfg):
    """Helper function to create dm_control environment"""
    domain, task = cfg.task_name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)

    env = dmc_POMDP.make_states_only(domain_name=domain,
                                    task_name=task,
                                    seed=cfg.seed,
                                    observables=cfg.observables,
                                    action_observable=cfg.action_observable,
                                    state_skip=cfg.state_skip,
                                    num_states = cfg.states_stack)
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

    def reset_episode(self, obs):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.float32))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(obs)

    def append(self, obs, action):
        self._state.append(obs)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state).reshape(1, -1)

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        print(f"Observables: {cfg.observables}, Action Observable: {cfg.action_observable}, From Dem: {cfg.from_dem}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        self.train_env = make_env(cfg)
        self.eval_env = make_env(cfg)

        if cfg.action_observable:
            obs_space = (self.cfg.states_stack*(self.train_env.observation_space.shape[0]+self.train_env.action_space.shape[0]),)

        else:
            obs_space = (self.cfg.states_stack*self.train_env.observation_space.shape[0],)

        cfg.agent.obs_shape = obs_space[0]
        cfg.agent.action_shape = self.train_env.action_space.shape[0]
        # cfg.agent.action_range = [float(self.env.action_space.low.min()),
        #                           float(self.env.action_space.high.max())]
        
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBufferAgent(self.cfg.replay_buffer_size, self.cfg.sequence_length, 
                                                obs_space, self.train_env.action_space.shape,
                                                self.device)

        #create source envs and agent
        self.expert_env = make_env(self.cfg)
        self.cfg.expert.obs_dim = self.expert_env.observation_space.shape[0]
        self.cfg.expert.action_dim = self.expert_env.action_space.shape[0]
        self.cfg.expert.action_range = [float(self.expert_env.action_space.low.min()),
                                        float(self.expert_env.action_space.high.max())]
        
        self.expert = hydra.utils.instantiate(self.cfg.expert)
        
        self.replay_buffer_expert = ReplayBufferExpert(self.cfg.replay_buffer_size, self.cfg.sequence_length, 
                                                        obs_space, self.train_env.action_space.shape,
                                                        self.device)

        #store video
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.ob = LMObservation(obs_space, self.train_env.action_space.shape, self.cfg.sequence_length)
        self.ob_test = LMObservation(obs_space, self.train_env.action_space.shape, self.cfg.sequence_length)
        
        self.timer = utils.Timer()
        self._step = 0
        self._global_episode = 0
        
    @property
    def step(self):
        return self._step

    @property
    def global_episode(self):
        return self._global_episode

    def store_expert_transitions(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
        
        while eval_until_episode(episode):
            obs, obs_partial, _ = self.expert_env.reset_full()
            self.expert.reset()
            self.video_recorder.init(enabled=(episode == 0))

            self.replay_buffer_expert.reset_episode(obs_partial)
            
            done = False
            episode_step = 0
            
            while not done:
                with torch.no_grad(), utils.eval_mode(self.expert):
                    action = self.expert.act(obs, self.step, eval_mode=True)
                obs, obs_partial, reward, done, _, _ = self.expert_env.step_full(action)    
                
                # allow infinite bootstrap
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.expert_env._max_episode_steps else done

                self.replay_buffer_expert.append(obs_partial, action, reward, done)                
                self.video_recorder.record(self.expert_env)
                
                total_reward += reward
                episode_step += 1
                step += 1

            episode += 1
            self.video_recorder.save('expert.mp4')

        print(f'Average expert reward: {total_reward / episode}, Total number of samples: {step}')

    def evaluate(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs, _ = self.eval_env.reset()
            self.ob_test.reset_episode(obs)
            self.video_recorder.init(enabled=(episode == 0))
            done = False

            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(self.ob_test, self.step, eval_mode=True)

                obs, reward, done, _, _ = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                self.ob_test.append(obs, action)

                total_reward += reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.step}.mp4')

        with self.logger.log_and_dump_ctx(self.step, ty='eval') as log:
            log('episode_reward', total_reward/episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        train_latent_only_until_step = utils.Until(self.cfg.num_seed_frames_update_latent_only, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        metrics = None
        obs, _ = self.train_env.reset()
        done = False

        self.replay_buffer.reset_episode(obs)
        self.ob.reset_episode(obs)

        while train_until_step(self.step):
            if done:
                self._global_episode += 1

                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.step, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.step)

                obs, _ = self.train_env.reset()
                self.replay_buffer.reset_episode(obs)
                self.ob.reset_episode(obs)

                done = False
                episode_reward = 0
                episode_step = 0

            # try to evaluate
            if eval_every_step(self.step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.step)
                self.evaluate()

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(self.ob, self.step, eval_mode=False)

            # run training update
            if not seed_until_step(self.step):

                if not train_latent_only_until_step(self.step):
                    metrics = self.agent.update(self.replay_buffer, self.replay_buffer_expert, self.step)
                    self.logger.log_metrics(metrics, self.step, ty='train')

                else:
                    _ = self.agent.update_latent(self.replay_buffer)

            obs, reward, done, _, _ = self.train_env.step(action)
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.train_env._max_episode_steps else done

            self.ob.append(obs, action)
            self.replay_buffer.append(obs, action, reward, done)

            episode_reward += reward
            episode_step += 1
            self._step += 1
            
    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        keys_to_save = ['agent', '_step', '_global_episode']
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


@hydra.main(config_path='config_folder/POMDP_no_visual', config_name='config_lail_POMDP')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    parent_dir = root_dir.parents[3]
    snapshot = parent_dir / f'expert_policies/snapshot_{cfg.task_name}_frame_skip_{cfg.state_skip}.pt'
    assert snapshot.exists()
    print(f'loading expert target: {snapshot}')
    workspace.load_expert(snapshot)
    workspace.store_expert_transitions()
    workspace.train()

if __name__ == '__main__':
    main()
