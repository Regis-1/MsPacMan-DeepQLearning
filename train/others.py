import os
import json
import logging

from logging.handlers import TimedRotatingFileHandler

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw

import numpy as np

def get_conf(save_dir):
    '''
    Will find and read json-formatted conf from save dir
    '''
    conf = None

    # find conf file in save_dir
    conf_count = 0
    for fn in os.listdir(save_dir):
        if fn.endswith('.conf'):
            with open(os.path.join(save_dir, fn)) as fin:
                conf = json.load(fin)
                conf_count += 1

    if conf_count > 1:
        raise Exception('too many confs in directory')

    return conf


logger = None

def init_logging(save_path=None, add_std_err=True):
    '''
    Initialize singleton style logger
    '''

    global logger
    if logger:
        return logger

    logger = logging.getLogger('rl')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime).19s [%(levelname)s] %(message)s')

    if save_path is not None:
        log_path = save_path + '.log'
        hdlr = TimedRotatingFileHandler(log_path, when='D')
        hdlr.setFormatter(formatter)

        logger.addHandler(hdlr)

    if add_std_err:
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    return logger


def log(*mesg):
    '''
    Log a message
    '''

    global logger

    if logger:
        logger.info(' '.join([str(m) for m in mesg]))
    else:
        print(' '.join([str(m) for m in mesg]))


def render_state(state, repeat=True, interval=60):
    '''
    Renders a state to the screen
    '''
    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,


    frames = []
    for i in range(state.shape[2]):
        # print(state[:,:,i].shape)
        shape = state[:,:,i].shape
        frame = state[:,:,i].reshape(shape[0], shape[1], 1)

        img_dat = np.concatenate((frame, frame, frame), axis=2)
        img_dat = Image.fromarray((img_dat * 255).astype('uint8'))
        draw = ImageDraw.Draw(img_dat)
        draw.text((0,0), "fr {}".format(i))

        frames.append(img_dat)

    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('on')

    video = animation.FuncAnimation(fig, 
                                    update_scene, 
                                    fargs=(frames, patch), 
                                    frames=len(frames), 
                                    repeat=repeat, 
                                    interval=interval)
    plt.show()

    return video


def render_game(game_frames, infos, render_func=None, repeat=False, interval=60, save_path=None, display=False):
    '''
    Renders a game to the screen or to disk
    '''    
    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,


    frames = []
    for i in range(len(game_frames)):
        if render_func:
            obs = render_func(game_frames[i])
        else:
            obs = game_frames[i]

        img_dat = Image.fromarray(obs)
        draw = ImageDraw.Draw(img_dat)

        info_stack = []
        for key, value in infos[i].items():
            info_stack.append('{}:{}'.format(key, value))
        info_str = ' '.join(info_stack)

        draw.text((0,0), "fr{} {}".format(i, info_str))

        frames.append(img_dat)


    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('on')

    video = animation.FuncAnimation(fig, 
                                    update_scene, 
                                    fargs=(frames, patch), 
                                    frames=len(frames), 
                                    repeat=repeat, 
                                    interval=interval)


    if display:
        plt.show()

    if save_path is not None:
        video.save(save_path)

    return video

