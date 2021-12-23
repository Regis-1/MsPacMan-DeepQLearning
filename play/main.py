from play_manager import PlayManager
from play_agent import PlayAgent
from play_env import PlayEnvironment

parameters = {
        'action_space': 9,
        'save_path_prefix': 'data/mspacman_4\\MsPacmanEnvironment',
        'tf_log_level': 3,
    }

def main():
    #create instance of PlayEnvironment
    env = PlayEnvironment()
    #create instance of PlayAgent
    agent = PlayAgent(parameters)

    with agent:
        #create instance of PlayManager for access to play method
        manager = PlayManager(parameters, env, agent)
        #activate play method with given fps and zoom attributes
        manager.play(fps=30, zoom=2)

    
if __name__ == '__main__':
    main()