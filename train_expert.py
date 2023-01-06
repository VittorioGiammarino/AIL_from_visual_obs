#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import time
import torch

from pathlib import Path

from video import VideoRecorder_Expert as VideoRecorder
from logger_folder.logger_expert import Logger
from buffers.replay_buffer_expert import ReplayBuffer
from utils_folder import utils as utils_expert

from DeepMind_control import dmc_expert
import hydra

def make_env(cfg):
    """Helper function to create dm_control environment"""
    domain, task = cfg.task_name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)

    env = dmc_expert.make_states_only(domain_name=domain,
                                    task_name=task,
                                    seed=cfg.seed,
                                    frame_skip=cfg.frame_skip)
    env.seed(cfg.seed)
    
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent="standard")

        utils_expert.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [float(self.env.action_space.low.min()),
                                  float(self.env.action_space.high.max())]
        
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        
        self._step = 0
        self._global_episode = 0
        
    @property
    def step(self):
        return self._step

    @property
    def global_episode(self):
        return self._global_episode

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with torch.no_grad(), utils_expert.eval_mode(self.agent):
                    action = self.agent.act(obs, self.step, eval_mode=True)
                obs, reward, done, _, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs, _ = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                self._global_episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad(), utils_expert.eval_mode(self.agent):
                    action = self.agent.act(obs, self.step, eval_mode=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self._step += 1
            
    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name}.pt'
        keys_to_save = ['agent', '_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)


@hydra.main(config_path='config_folder/MDP', config_name='config_RL_only')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
