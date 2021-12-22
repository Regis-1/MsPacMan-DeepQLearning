import sys
import os
import gym
import pygame

from environment import Environment
from agent import Agent
from manager import Manager

from others import init_logging

train = True

#data/mspacman_1
parameters = {
        'environment': 'MsPacmanEnvironment',
        'model_environment': 'MsPacman',
        'model_save_prefix': 'data/mspacman_3\MsPacmanEnvironment',
        'save_dir': 'data/mspacman_3',
        'max_num_training_steps': 10000000,
        'replay_max_memory_length': 1000000,
        'action_space': 9,
        'interval': 60,
        'use_epsilon': False,
        'num_games': None,
        'frame_skip': 1,
        'use_episodes': True,
        'max_game_length': 50000,
        'is_training': True,
        'display': False,
        'save_video': False,
        'num_frames_per_state': 4,
        'action_space': 9,
        'save_path_prefix': 'data/mspacman_3\\MsPacmanEnvironment'
    }

parameters['is_training'] = train

env = Environment()
agent = Agent(parameters)
init_logging(save_path='data/mspacman_3\\MsPacmanEnvironment')

with agent:
    manager = Manager(parameters, env, agent)
    if train:
        manager.run()
    else:
        manager.play(fps=25, zoom=2)