import sys
import os
import gym
import pygame

from environment import Environment
from agent import Agent
from manager import Manager

from others import init_logging

parameters = {
        'action_space': 9,
        'interval': 60,
        'use_epsilon': False,
        'num_games': None,
        'display': False,
        'save_video': False,
        'num_frames_per_state': 4,
        'save_path_prefix': 'data/mspacman_3\\MsPacmanEnvironment',
        #agent
        'save_dir': 'data/mspacman_3',
        'game_id': None,
        'eps_min': 0.1,
        'eps_max': 1.0,
        'eps_decay_steps': 2000000,
        'save_model_steps': 10000,
        'copy_network_steps': 10000,
        'batch_size': 32,
        'model_save_prefix': 'data/mspacman_3\MsPacmanEnvironment',
        'replay_max_memory_length': 1_000_000,
        'replay_cache_size': 300000,
        'max_num_training_steps': 6_000_000,
        'num_game_frames_before_training': 10000,
        'num_game_steps_per_train': 4,
        'num_train_steps_save_video': None,
        'train_report_interval': 100,
        'use_episodes': True,
        'use_log': True,
        'frame_skip': 1,
        'max_game_length': 50000,
        'tf_log_level': 3,
        'per_a': 0.6,
        'per_b_start': 0.4,
        'per_b_end': 1,
        'per_anneal_steps': 2000000,
        'per_calculate_steps': 5000,
        #model
        'learning_rate': 0.00025,
        'discount_rate': 0.99,
        'is_training': True
    }

env = Environment()
agent = Agent(parameters)
init_logging(save_path='data/mspacman_3\\MsPacmanEnvironment')

with agent:
    manager = Manager(parameters, env, agent)
    manager.run()