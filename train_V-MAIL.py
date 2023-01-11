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

import hydra
import functools
import numpy as np
import torch
from dm_env import specs

from DeepMind_control import dmc_dreamer
from DeepMind_control.dmc_expert import _spec_to_box
from utils_folder import utils_dreamer as utils
from logger_folder.logger import Logger
from video import TrainVideoRecorder, VideoRecorder

from agents.dreamerv2_w_expert_obs_adv import DreamerV2Agent
from agents.dreamerv2_w_expert_obs_only_adv import DreamerV2Agent_IL_from_obs_only

torch.backends.cudnn.benchmark = True

def make_dataset(episodes, config):
  generator = utils.sample_episodes(episodes, config.batch_length_training, config.oversample_ends)
  dataset = utils.from_generator(generator, config.batch_size_training)
  return dataset

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.setup()

        self.train_dataset = make_dataset(self.train_eps, self.cfg)
        self.eval_dataset = make_dataset(self.eval_eps, self.cfg)
        self.expert_dataset = make_dataset(self.expert_eps, self.cfg)

        if cfg.IL_from_obs:
            print("Learning from expert states only")
            self.cfg.batch_length_training += 1
            self.agent = DreamerV2Agent_IL_from_obs_only(self.train_env.action_space.shape, 
                                                        self.cfg.device, self.train_dataset, self.traindir, 
                                                        self.expert_dataset, self.expertdir,
                                                        self.cfg).to(self.cfg.device)
        else:
            print("Learning from expert states and actions")
            self.agent = DreamerV2Agent(self.train_env.action_space.shape, 
                                        self.cfg.device, self.train_dataset, self.traindir, 
                                        self.expert_dataset, self.expertdir,
                                        self.cfg).to(self.cfg.device)

        self.agent.requires_grad_(requires_grad=False)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        self.traindir = self.work_dir / 'train_eps'
        self.traindir.mkdir(parents=True, exist_ok=True)

        self.evaldir = self.work_dir / 'eval_eps'
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.expertdir = self.work_dir / 'expert_eps'
        self.expertdir.mkdir(parents=True, exist_ok=True)

        print('Create envs')
        self.train_eps = utils.load_episodes(self.traindir, limit=self.cfg.dataset_size)
        self.eval_eps = utils.load_episodes(self.evaldir, limit=1)
        self.expert_eps = utils.load_episodes(self.expertdir, limit=self.cfg.dataset_size)

        self.train_env = dmc_dreamer.make_env(self.cfg, self.traindir, self.evaldir, 'train', self.train_eps, self.eval_eps)
        self.eval_env = dmc_dreamer.make_env(self.cfg, self.traindir, self.evaldir, 'eval', self.train_eps, self.eval_eps)
        self.expert_env = dmc_dreamer.make_env_expert(self.cfg, self.expertdir, self.expert_eps)
        
        self.cfg.expert.obs_dim = self.expert_env.observation_space['state'].shape[0]
        self.cfg.expert.action_dim = self.expert_env.action_space.shape[0]
        self.cfg.expert.action_range = [float(self.expert_env.action_space.low.min()),
                                        float(self.expert_env.action_space.high.max())]
        
        self.expert = hydra.utils.instantiate(self.cfg.expert)

        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def store_expert_dataset(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_expert_episodes)

        while eval_until_episode(episode):
            obs = self.expert_env.reset()
            done = False
            self.video_recorder.init(self.expert_env, enabled=(episode == 0))

            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.expert.act(obs['state'], sample=True)
                
                obs, reward, done, _ = self.expert_env.step(action)
                self.video_recorder.record(self.expert_env)
                total_reward += reward
                step += 1

            episode += 1
            self.video_recorder.save('expert.mp4')

        print(f'Average expert reward: {total_reward / episode}, Total number of samples: {step}')

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs = self.eval_env.reset()
            done = False
            agent_state = None
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))

            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, agent_state = self.agent.act(obs, agent_state, self.global_step, eval_mode=True)
                
                obs, reward, done, _ = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += reward
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
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)
        should_pretrain = utils.Once()

        episode_step, episode_reward = 0, 0
        obs = self.train_env.reset()
        done = False
        agent_state = None
        self.train_video_recorder.init(obs['image'])
        metrics = None

        while train_until_step(self.global_step):
            if done:
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('step', self.global_step)

                # reset env
                obs = self.train_env.reset()
                done = False
                agent_state = None
                self.train_video_recorder.init(obs['image'])
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
                action, agent_state = self.agent.act(obs, agent_state, 
                                                    self.global_step, eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):

                if should_pretrain():
                    metrics = self.agent.pretrain(self.cfg.pretrain)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

                metrics = self.agent.update(self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            obs, reward, done, _  = self.train_env.step(action)
            episode_reward += reward
            self.train_video_recorder.record(obs["image"])
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def load_expert(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.expert = payload['agent']


@hydra.main(config_path='config_folder/POMDP', config_name='config_dreamer_obs_adv')
def main(cfg):
    from train_dreamer_w_expert_obs_adv import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    parent_dir = root_dir.parents[3]
    snapshot = parent_dir / f'expert_source_policies/snapshot_{cfg.task_name}.pt'
    assert snapshot.exists()
    print(f'loading expert target: {snapshot}')
    workspace.load_expert(snapshot)
    workspace.store_expert_dataset()
    workspace.train()

if __name__ == '__main__':
    main()