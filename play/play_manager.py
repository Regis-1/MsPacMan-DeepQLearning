import time
import numpy as np
from collections import deque

import pygame
pygame.font.init()
from pygame.locals import VIDEORESIZE

from play_agent import PlayAgent

#constants made for pygame font system
font = pygame.font.SysFont('Arial, Times New Roman', 24)
WHITE_COLOR = (255,255,255)
STATE_TEXT = (font.render('HUMAN', True, WHITE_COLOR), font.render('AI', True, WHITE_COLOR))

#function needed for play() method
def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))

class PlayManager:

    RESET_GAME_STATE = {
        'game_start_time': time.time(),
        'game_done': False,
        'episode_done': False,
        'total_max_q': 0,
        'iteration': 0,
        'game_length': 0,
        'episode_length': 0,
        'score': 0,
        'observation': None,
        'old_state': None,
        'state': None,
        'action': 0,
        'reward': None,
        'cont': 1,
        'info': None,
        'num_lives': None,
        'frames_render': [],
        'infos_render': [],
    }   


    def __init__(self, parameters, env, agent):
        self.env = env
        self.agent = agent

        # set up default values
        self.parameters = parameters
        self.game_state = {}
    
    #play game with option to switch between manual and AI controls
    def play(self, transpose=True, fps=20, zoom=3):
        self.num_frames_per_state = 4
        self.game_state['state'] = 0
        keys_to_action = 0
        action = 0
        play_state = 0
        
        border_color = (255,0,0)
        
        self.game_state['observation'] = self.env.reset()
        self.game_state['frames'] = deque(maxlen=self.num_frames_per_state)
        self.game_state['frames'].append(self.game_state['observation'])
        
        rendered = self.env.env.render(mode="rgb_array")
        
        #get keys mapping for MsPacMan environment
        if hasattr(self.env.env, "get_keys_to_action"):
            keys_to_action = self.env.env.get_keys_to_action()
        elif hasattr(self.env.env.unwrapped, "get_keys_to_action"):
            keys_to_action = self.env.env.unwrapped.get_keys_to_action()

        relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

        video_size = [rendered.shape[1], rendered.shape[0]]
        if zoom is not None:
            video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

        pressed_keys = []
        running = True
        env_done = True

        screen = pygame.display.set_mode(video_size)
        clock = pygame.time.Clock()

        #MAIN GAME LOOP
        while running:
            
            if env_done:
                env_done = False
                obs = self.env.reset()
            else:
                if play_state == 0:
                    #manual action
                    action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                else:
                    #AI action choice
                    action = self.agent.get_action(self.game_state['state'])

            obs, rew, env_done, info = self.env.step(action)        
            self.game_state['frames'].append(obs)
            
            if len(self.game_state['frames']) >= self.num_frames_per_state:
                self.game_state['old_state'] = self.game_state['state']
                self.game_state['state'] = np.concatenate(self.game_state['frames'], axis=2)

            #render frame
            if obs is not None:
                rendered = self.env.env.render(mode="rgb_array")
                display_arr(screen, rendered, transpose=transpose, video_size=video_size)

            # process pygame events
            for event in pygame.event.get():
                # test events, set key states
                if event.type == pygame.KEYDOWN:
                    if event.key in relevant_keys:
                        pressed_keys.append(event.key)
                    elif event.key == 27:
                        running = False
                    elif event.key == ord('f'):
                        play_state = int(not bool(play_state))
                    elif event.key == ord('r'):
                        env_done = True
                elif event.type == pygame.KEYUP:
                    if event.key in relevant_keys:
                        pressed_keys.remove(event.key)
                elif event.type == pygame.QUIT:
                    running = False
                elif event.type == VIDEORESIZE:
                    video_size = event.size
                    screen = pygame.display.set_mode(video_size)
                    print(video_size)

            #add UI overlay to inform about choosen mode (manual or AI)
            screen.blit(STATE_TEXT[play_state], (5,video_size[1]-40))
            if bool(play_state):
                pygame.draw.rect(screen, border_color, pygame.Rect(1, 1, video_size[0]-2, video_size[1]-2),  4)
            pygame.display.flip()
            clock.tick(fps)
        pygame.quit()