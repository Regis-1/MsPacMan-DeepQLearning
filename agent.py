import logging
import importlib
import inspect
import os
import time

from collections import deque

import gym
import tensorflow as tf
import numpy as np

from model import Model
from memory import ReplayMemory, ReplayMemoryDisk, ReplaySamplerPriority

from others import init_logging, log

class Agent:
    parameters = {
        'is_training': True,
        'save_dir': './data',
        'game_id': None,
        'eps_min': 0.1,
        'eps_max': 1.0,
        'eps_decay_steps': 2000000,
        'discount_rate': 0.99,
        'save_model_steps': 10000,
        'copy_network_steps': 10000,
        'batch_size': 32,
        'model_save_prefix': None,
        'replay_max_memory_length': 300000,
        'replay_cache_size': 300000,
        'max_num_training_steps': 2000000,
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
        'per_calculate_steps': 5000 }
        
    MAX_MEMORY_BATCH_SIZE = 128
    MIN_ERROR_PRIORITY = 0.01
    MAX_ERROR_PRIORITY = 1.0
    
    def __init__(self, params):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.parameters['tf_log_level']) #reduce log out for tensorflow
        self.parameters.update(params)
        self.model = Model(self.parameters)
        self.max_num_training_steps = self.parameters['max_num_training_steps']
        self.replay_max_memory_length = self.parameters['replay_max_memory_length']
        self.num_game_frames_before_training = self.parameters['num_game_frames_before_training']
        self.batch_size = self.parameters['batch_size']
        self.save_steps = self.parameters['save_model_steps']
        self.copy_steps = self.parameters['copy_network_steps']
        self.train_report_interval = self.parameters['train_report_interval']
        self.num_game_steps_per_train = self.parameters['num_game_steps_per_train']
        self.use_per = self.parameters['use_per']
        self.num_train_steps_save_video = self.parameters['num_train_steps_save_video']

        self.eps_min = self.parameters['eps_min']
        self.eps_max = self.parameters['eps_max']
        self.eps_decay_steps = self.parameters['eps_decay_steps']
        self.is_training = self.parameters['is_training']
        self.save_path_prefix = self.parameters['save_path_prefix']
        
    def train(self, game_state):
        if not self.is_training:
            return True

        if self.is_training_done():
            return False

        if game_state['game_done']:
            self.model.set_game_count(self.model.get_game_count() + 1)

        if game_state['old_state'] is not None:
            self._add_memories(state=game_state['old_state'], 
                               action=game_state['action'], 
                               reward=game_state['reward'], 
                               cont=game_state['cont'], 
                               next_state=game_state['state'])                

        # add X amount of replay memory before starting training
        if len(self.replay_sampler) < self.num_game_frames_before_training:
            return True

        if self.game_step_count == 0:
            log('[train] train start')

        # only train every N steps
        self.game_step_count += 1
        if self.game_step_count % self.num_game_steps_per_train != 0:
            return True        

        self._train_step()

        return True
        
    def is_training_done(self):
        return self.step >= self.max_num_training_steps
        
    def __enter__(self):
        self.model.open()

        if self.is_training:
            self._train_init()
        else:
            self._play_init()

    def _train_init(self):
        self.train_start_time = time.time()
        self.game_step_count = 0
        self.step = self.model.get_training_step()

        self.train_report_start_time = time.time()
        self.train_report_last_step = self.step
        self.total_losses = []
        
        self._train_init_memory()

    def _init_per(self):
        self.per_memory_batch = ReplayMemory(self.model.INPUT_HEIGHT,
                                                self.model.INPUT_WIDTH,
                                                self.model.INPUT_CHANNELS,
                                                max_size=self.MAX_MEMORY_BATCH_SIZE,
                                                state_type=self.model.STATE_TYPE)

        self.replay_sampler = ReplaySamplerPriority(self.memories)

        self.per_a = self.parameters['per_a']

        self.per_b_start = self.parameters['per_b_start']
        self.per_b_end = self.parameters['per_b_end']
        self.per_b = self.per_b_start

        self.per_anneal_steps = self.parameters['per_anneal_steps']
        self.per_calculate_steps = self.parameters['per_calculate_steps']

        self.last_min_loss = None
        self.last_max_loss = None
        self.last_max_weight = None

        self.tree_idxes = np.zeros((self.batch_size), dtype=int)
        self.priorities = np.zeros((self.batch_size), dtype=float)

    def _train_init_memory(self):
        # training batch
        self.train_batch = ReplayMemory(self.model.INPUT_HEIGHT,
                                        self.model.INPUT_WIDTH,
                                        self.model.INPUT_CHANNELS,
                                        max_size=self.batch_size,
                                        state_type=self.model.STATE_TYPE)


        self.memories = ReplayMemoryDisk(self._get_replay_memory_path(),
                                         self.model.INPUT_HEIGHT,
                                         self.model.INPUT_WIDTH,
                                         self.model.INPUT_CHANNELS,
                                         state_type=self.model.STATE_TYPE,
                                         max_size=self.replay_max_memory_length,
                                         cache_size=self.parameters['replay_cache_size'])

        self._init_per()
        
    def _train_step(self):
        #ACTUAL TRAINING
        # sample        
        self._sample_memories()

        # train the model
        self.step, losses, loss = self.model.train(self.train_batch.states,
                                                   self.train_batch.actions,
                                                   self.train_batch.rewards,
                                                   self.train_batch.continues,
                                                   self.train_batch.next_states,
                                                #    target_max_q_values,
                                                   is_weights=self.is_weights)

        # update priority steps in sum tree
        losses = self._make_priority(losses)
        self.replay_sampler.update_sum_tree(self.tree_idxes, losses)

        self.total_losses.append(loss)
        
        #SAVE MODEL
        if self.step % self.copy_steps == 0:
            log('copying online to target dqn')
            self.model.copy_network()


        # And save regularly
        if self.step % self.save_steps == 0:
            self.model.save(self.save_path_prefix)
            
        #REPORT STATS
        if self.step % self.train_report_interval == 0:
            elapsed = time.time() - self.train_report_start_time
            if elapsed > 0:
                frame_rate = (self.step - self.train_report_last_step) / elapsed
            else:
                frame_rate = 0.0

            self.train_report_last_step = self.step
            self.train_report_start_time = time.time()

            if len(self.total_losses) > 0:
                avg_loss = sum(self.total_losses) / len(self.total_losses)
            else:
                avg_loss = 0

            self.total_losses.clear()

            per_str = 'per_ab: {:0.2f}/{:0.2f}'.format(self.per_a, self.per_b) + \
                      ' avg/min/max loss: {:0.3f}/{:0.3f}/{:0.3f}'.format(self.last_avg_loss, self.last_min_loss, self.last_max_loss) + \
                      ' max_weight: {:0.3f} sum total: {:0.1f} '.format(self.last_max_weight, self.replay_sampler.total)

            log('[train] step {} game {}/{}'.format(self.step,
                                                    self.game_step_count,
                                                    self.model.get_game_count()) + \
                ' avg loss: {:0.5f} {}mem: {:d} '.format(avg_loss,
                                                         per_str, 
                                                         len(self.replay_sampler)) + \
                ' fr: {:0.1f} eps: {:0.2f} '.format(frame_rate,
                                                    self._get_epsilon(self.step)))

    def _train_finish(self):
        log('train finish')
        log('closing replay memory')

        self.replay_sampler.close()

        # save game count
        # self.model.set_game_count(self.game_runner.total_game_count)
        self.model.save(self.save_path_prefix)

        elapsed = time.time() - self.train_start_time

        log('train finished in {:0.1f} seconds / {:0.1f} mins / {:0.1f} hours'.format(elapsed, elapsed / 60, elapsed / 60 / 60))
        
    def _add_memories(self, state, action, reward, cont, next_state):
        if self.use_per:
            self._add_priority_memory(state, action, reward, cont, next_state)
        else:
            self.replay_sampler.append(state=state,
                                       action=action,
                                       reward=reward,
                                       next_state=next_state,
                                       cont=cont)

    def _add_priority_memory(self, state, action, reward, cont, next_state):
        self.per_memory_batch.append(state=state,
                                     action=action,
                                     reward=reward,
                                     cont=cont,
                                     next_state=next_state)

        # append only every after MAX_MEMORY_BATCH_SIZE memories are added
        if len(self.per_memory_batch) >= self.MAX_MEMORY_BATCH_SIZE:

            # calculate losses
            losses = self.model.get_losses(self.per_memory_batch.states,
                                           self.per_memory_batch.actions,
                                           self.per_memory_batch.rewards,
                                           self.per_memory_batch.continues,
                                           self.per_memory_batch.next_states)

            losses = self._make_priority(losses)

            for i in range(len(self.per_memory_batch)):
                self.replay_sampler.append(state=self.per_memory_batch.states[i],
                                           action=self.per_memory_batch.actions[i],
                                           reward=self.per_memory_batch.rewards[i],
                                           next_state=self.per_memory_batch.next_states[i],
                                           cont=self.per_memory_batch.continues[i],
                                           loss=losses[i])
            self.per_memory_batch.clear()

    def _sample_memories(self):
        if self.use_per:
            if self.step % self.per_calculate_steps == 0 or self.last_max_weight is None:
                self.last_avg_loss = self.replay_sampler.get_average()
                self.last_min_loss = max(self.replay_sampler.get_min(), self.MIN_ERROR_PRIORITY)
                self.last_max_loss = min(self.replay_sampler.get_max(), self.MAX_ERROR_PRIORITY)
                self.last_max_weight = pow(self.batch_size * (self.last_min_loss / self.replay_sampler.total), -self.per_b)

            # sample memories from sum tree
            self.replay_sampler.sample_memories(self.train_batch,
                                                batch_size=self.batch_size,
                                                tree_idxes=self.tree_idxes,
                                                priorities=self.priorities)

            sampling_probs = self.priorities / self.replay_sampler.total
            self.is_weights = np.power(self.batch_size * sampling_probs, -self.per_b) / self.last_max_weight 
        else:
            # sample randomly from each range
            self.replay_sampler.sample_memories(self.train_batch,
                                                batch_size=self.batch_size)
            self.is_weights = None

    def _make_priority(self, losses):
        return np.power(np.minimum(losses + self.MIN_ERROR_PRIORITY, 
                                   self.MAX_ERROR_PRIORITY), 
                        self.per_a)

    def _get_replay_memory_path(self):
        return '{}_replay_memory.hdf5'.format(self.save_path_prefix)

    def _play_init(self):
        self.use_epsilon = self.parameters['use_epsilon']

    def _play_finish(self):
        pass

    def get_action(self, state):
        q_values = self.model.predict([state])

        if self.is_training or self.use_epsilon:
            return self._epsilon_greedy(q_values, self.step)
        else:
            return np.argmax(q_values)

    def _get_epsilon(self, step):
        '''
        Gets current epsilon based on what step and the epsilon range
        '''
        return max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)


    def _epsilon_greedy(self, q_values, step):
        '''
        Returns the optimal value if over epsilon, other wise returns the argmax action
        '''
        epsilon = self._get_epsilon(step)

        if np.random.rand() < epsilon:
            # random action
            return np.random.randint(self.model.num_outputs) 
        else:
            # optimal action
            return np.argmax(q_values)

    def open(self):
        self.model.__enter__()

        if self.is_training:
            self._train_init()
        else:
            self._play_init()


    def close(self, ty=None, value=None, tb=None):
        if self.is_training:
            self._train_finish()
        else:
            self._play_finish()

        self.model.__exit__(ty, value, tb)


    def __enter__(self):
        return self.open()


    def __exit__(self, ty, value, tb):
        self.close(ty, value, tb)