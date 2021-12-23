import time

import gym
import numpy as np


class Environment:
    GAME_ID = 'MsPacmanDeterministic-v4'
    COMPRESS_RATIO = 2
    GAME_REPORT_INTERVAL = 10
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80

    ACTION_NOTHING = 0

    num_lives = 0

    def __init__(self):
        self.env = gym.make(self.GAME_ID)

        self.env.seed(int(time.time()))

        self.raw_obs = None
        self.reward = None
        self.done = None
        self.info = None


    def get_action_space(self):
        return self.env.action_space.n


    def step(self, action):
        action = self.before_action(
            action, self.raw_obs, self.reward, self.done, self.info)

        self.raw_obs, self.reward, self.done, self.info = self.env.step(action)

        return self.preprocess_observation(self.raw_obs), self.reward, self.done, self.info


    def reset(self):
        return self.preprocess_observation(self.env.reset())


    def preprocess_observation(self, obs):
        # cut off score and icons from bottom
        obs = obs[0:-38:self.COMPRESS_RATIO,
                  ::self.COMPRESS_RATIO]  # crop and downsize
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.144])

        return obs.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


    def before_action(self, action, obs, reward, done, info):
        # skip time in intro and between episodes
        if info is not None and info['ale.lives'] != self.num_lives:
            self.num_lives = info['ale.lives']

            action = self.ACTION_NOTHING

            for _ in range(30):
                self.env.step(self.ACTION_NOTHING)

        return action
