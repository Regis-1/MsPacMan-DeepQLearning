import os
import importlib
import gym
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from collections import deque
import numpy as np

from PIL import Image, ImageDraw
from others import log

from memory import ReplayMemory, ReplayMemoryDisk, ReplaySamplerPriority

class Model:
    
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4
    STATE_TYPE = 'uint8'
    
    def __init__(self, 
                 parameters):
        self.parameters = parameters
        self.init_model = True
        self.load_model = True
        self.save_model = False
        
        self.num_outputs = self.parameters['action_space']
        self.discount_rate = self.parameters['discount_rate']
        
    def __enter__(self):
        log('creating new session load_model:', self.load_model, 'save_model:', self.save_model)

        tf.compat.v1.reset_default_graph()

        if self.init_model:
            self._make_inputs()
            self._make_network()
            self._make_train()

        self._init_tf()

        return self

    def _init_tf(self):
        # make saver
        self.saver = tf.compat.v1.train.Saver()

        # set tensorflow to NOT use all of GPU memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create actual tf session
        self._sess = tf.compat.v1.Session(config=config)
        self._sess.as_default()
        self._sess.__enter__()

        loaded = False
        if self.load_model:
            loaded = self.restore(self.parameters['save_path_prefix'])

        if not loaded and not self.init_model:
            raise Exception('cannot load existing model')

        if not loaded:
            self._sess.run(tf.compat.v1.global_variables_initializer())

        tf.compat.v1.get_default_graph().finalize()
    
    def get_training_step(self):
        return self.training_step.eval()
        
    def predict(self, X_states, use_target=False):
        if use_target:
            q_values = self.target_q_values
        else:
            q_values = self.online_q_values

        values = self.run([q_values],
                          feed_dict={self.X_state: X_states})

        return values[0]
    
    def get_action(self, X_states, use_target=False):
        values = self.predict(X_states, use_target=use_target)

        return np.argmax(values, axis=1)

    def train(self, X_states, X_actions, rewards, continues, next_states, is_weights=None):
        target_q_values = self._get_target_q_values(rewards, continues, next_states)

        feed_dict = {
            self.X_state: X_states,
            self.X_action: X_actions,
            self.y: target_q_values,
            self.is_weights: is_weights
        }

        step, _, losses, loss = self.run([self.training_step,
                                          self.training_op,
                                          self.losses,
                                          self.loss],
                                          feed_dict=feed_dict)
        return step, losses, loss

    def get_losses(self, X_states, actions, rewards, continues, next_states):
        target_q_values = self._get_target_q_values(rewards, continues, next_states)


        return self.run([self.losses],
                        feed_dict={
                            self.X_state: X_states,
                            self.X_action: actions,
                            self.y: target_q_values
                        })[0]

    def _get_target_q_values(self, rewards, continues, next_states):
        max_q_values = self.run([self.max_q_values],
                                feed_dict={self.X_state: next_states})[0]

        return rewards + continues * self.discount_rate * max_q_values

    def copy_network(self):
        self.copy_online_to_target.run()        


    def get_training_step(self):
        return self.training_step.eval()
        

    def set_game_count(self, count):
        self.game_count.load(count)


    def get_game_count(self):
        return self.run([self.game_count])[0]

    def __exit__(self, ty, value, tb):
        if self.save_model:
            self.save(self.parameters['save_path_prefix'])

        self._sess.__exit__(ty, value, tb)                    
        self._sess.close()
        self._sess = None

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)
        
    def save(self, save_path_prefix):
        log('saving model: ', save_path_prefix)
        self.saver.save(self._sess, save_path_prefix)
        log('saved model')

    def restore(self, save_path_prefix):
        if not os.path.exists(save_path_prefix + '.index'):
            log('  model does not exist:', save_path_prefix)
            return False

        log('  restoring model: ', save_path_prefix)
        self.saver.restore(self._sess, save_path_prefix)
        log('  restored model')

        return True
        
    def _make_inputs(self):
        self.X_action = tf.compat.v1.placeholder(tf.uint8, shape=[None], name='action')

        # target Q
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None], name='y')

        self.is_weights = tf.compat.v1.placeholder(tf.float32, [None], name='is_weights')

        # save how many games we've played
        self.game_count = tf.Variable(0, trainable=False, name='game_count')

        # regular image input
        self.X_state = tf.compat.v1.placeholder(tf.uint8, shape=[None, 
                                                       self.INPUT_HEIGHT,
                                                       self.INPUT_WIDTH,
                                                       self.INPUT_CHANNELS])
        # convert rgb int (0-255) to floats
        last = tf.cast(self.X_state, tf.float32)
        self.input = tf.divide(last, 255)

    def _make_network(self):
        # make online and target q networks
        self.online_q_values, self.online_actions, online_vars = self._make_q_network(self.input, name='q_networks/online')
        self.target_q_values, self.target_actions, target_vars = self._make_q_network(self.input, name='q_networks/target')

        # use online to select action and target to get max q value
        self.max_q_values = tf.reduce_max(input_tensor=self.target_q_values * tf.one_hot(tf.argmax(input=self.online_q_values, axis=1), self.num_outputs), axis=1)


        # make copy settings action
        copy_ops = []
        for var_name, target_var in target_vars.items():
            copy_ops.append(target_var.assign(online_vars[var_name]))

        self.copy_online_to_target = tf.group(*copy_ops)


    def _make_q_network(self, X_input, name):
        last = X_input

        hidden_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)

        # make convolutional network 
        conv_num_maps = [32, 64, 64]
        # conv_kernel_sizes = [8, 4, 3]
        conv_kernel_sizes = [8, 4, 4]
        conv_strides = [4, 2, 1]
        conv_paddings = ['same'] * 3
        conv_activations = [tf.nn.relu] * 3
        num_hidden = 512

        with tf.compat.v1.variable_scope(name) as scope:
            # conv layers
            for num_maps, kernel_size, stride, padding, act_func in zip(conv_num_maps,
                                                                        conv_kernel_sizes,
                                                                        conv_strides,
                                                                        conv_paddings,
                                                                        conv_activations):
                last = tf.compat.v1.layers.conv2d(last, 
                                        num_maps, 
                                        kernel_size=kernel_size, 
                                        strides=stride, 
                                        padding=padding,
                                        activation=act_func)

            input_layer = tf.compat.v1.layers.flatten(last)

            # action output
            last = tf.compat.v1.layers.dense(input_layer,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            advantage = tf.compat.v1.layers.dense(last, 
                                        self.num_outputs,
                                        kernel_initializer=hidden_initializer)

            # value output
            last = tf.compat.v1.layers.dense(input_layer,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            value = tf.compat.v1.layers.dense(last, 
                                    1,
                                    kernel_initializer=hidden_initializer)

            # combine
            outputs = value + tf.subtract(advantage, tf.reduce_mean(input_tensor=advantage, 
                                                                    axis=1, 
                                                                    keepdims=True))

            actions = tf.argmax(input=outputs, axis=1)


        # get var names for copy
        var_dict = {}
        for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope.name):
            var_dict[var.name[len(scope.name):]] = var

        return outputs, actions, var_dict



    def _make_train(self):
        with tf.compat.v1.variable_scope("train"):
            self.online_max_q_values = tf.reduce_sum(input_tensor=self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float32),
                                          axis=1)
            self.abs_losses = tf.abs(self.y - self.online_max_q_values,
                                     name='abs_losses')

            self.loss = tf.reduce_mean(input_tensor=self.is_weights * tf.math.squared_difference(self.y, self.online_max_q_values))
            self.losses = self.abs_losses

            self.training_step = tf.Variable(0, 
                                    trainable=False, 
                                    name='step')

            optimizer = tf.compat.v1.train.AdamOptimizer(self.parameters['learning_rate'])

            self.training_op = optimizer.minimize(self.loss, 
                                                  global_step=self.training_step)