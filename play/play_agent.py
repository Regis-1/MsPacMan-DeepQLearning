import os
import numpy as np

from play_model import PlayModel

class PlayAgent:
    
    def __init__(self, params):
        self.parameters = params
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.parameters['tf_log_level']) #reduce log out for tensorflow

        self.model = PlayModel(self.parameters)
        
    def get_action(self, state):
        q_values = self.model.predict([state])

        return np.argmax(q_values)
        
    def __enter__(self):
        self.model.__enter__()

    def __exit__(self, ty, value, tb):
        pass