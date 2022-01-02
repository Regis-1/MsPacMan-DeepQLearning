import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os

from others import log

class PlayModel:
    
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4
    STATE_TYPE = 'uint8'
    
    def __init__(self, 
                 parameters):
        self.parameters = parameters
        self.init_model = True
        self.load_model = True
        
        self.num_outputs = self.parameters['action_space']
        
    def __enter__(self):
        log('creating new session load_model:', self.load_model)

        tf.compat.v1.reset_default_graph()

        self._make_inputs()
        self._make_network()

        self._init_tf()

        return self
        
    def _make_inputs(self):
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

            last = tf.compat.v1.layers.dense(input_layer,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            outputs = tf.compat.v1.layers.dense(last, 
                                      self.num_outputs,
                                      kernel_initializer=hidden_initializer)

            actions = tf.argmax(input=outputs, axis=1)


        # get var names for copy
        var_dict = {}
        for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope.name):
            var_dict[var.name[len(scope.name):]] = var

        return outputs, actions, var_dict
    
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
        
    def restore(self, save_path_prefix):
        if not os.path.exists(save_path_prefix + '.index'):
            log('  model does not exist:', save_path_prefix)
            return False

        log('  restoring model: ', save_path_prefix)
        self.saver.restore(self._sess, save_path_prefix)
        log('  restored model')

        return True
    
    def predict(self, X_states, use_target=False):
        if use_target:
            q_values = self.target_q_values
        else:
            q_values = self.online_q_values

        values = self.run([q_values],
                          feed_dict={self.X_state: X_states})

        return values[0]
    
    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)