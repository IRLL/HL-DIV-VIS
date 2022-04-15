"""
    A new file to build out options for the highlights code base.
"""

import tensorflow as tf
import argparse
import coloredlogs, logging
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
import stream_generator as stream_generator
import overlay_stream as overlay_stream
import video_generation as video_generation

import joblib
import os
import numpy as np
from datetime import datetime

#From de la cruz
def main():

    #get rid of distracting TF errors
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')

    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-env', type=str, default='MsPacmanNoFrameskip-v4', help='OpenAi Gym environment ID')
    parser.add_argument('--agent-model', type=str, default='MsPacman_5M_ingame_reward.h5', help='Name of model saved in model folder which you want to use, including h5 extension')

    parser.add_argument('--cuda-devices', type=str, default='')
    parser.add_argument('--gpu-fraction', type=float, default=0.333)
    parser.add_argument('--cpu-only', action='store_true')
    parser.set_defaults(cpu_only=False)

    #Britt additions
    parser.add_argument('--convert-model', action='store_true', help='run the algorithm to convert a tensorflow model to keras model')
    parser.set_defaults(convert_model=False)
    parser.add_argument('--generate-stream', action='store_true', help='output a stream into a new folder')
    parser.set_defaults(generate_stream=False)
    parser.add_argument('--overlay-saliency', action='store_true', help='use to overlay a saliency map onto the screenshots')
    parser.set_defaults(overlay_saliency=False)
    
    # get current date and time for default data output folder name
    # datetime object containing current date and time
    now = datetime.now()
     
    print("now =", now)

    # month_day_YY_H_M_S
    dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
    print("date and time =", dt_string)
    
    parser.add_argument('--stream-folder', type=str, default=dt_string)
    parser.add_argument('--generate-video', action='store_true', help='creates video of the summary states and screenshots with saliency maps')
    parser.add_argument('--minigrid', action='store_true', help='Use the minigrid environment')
    parser.set_defaults(minigrid=False)
    parser.add_argument('--num-steps', type=int, default=5)
    parser.add_argument('--watch-agent', action='store_true', help='shows a window with the agent acting in real-time')
    parser.set_defaults(watch_agent=False)
    parser.add_argument('--vis', action='store_true', help='generate additional plots and charts')
    parser.set_defaults(vis=False)
    parser.add_argument('--verbose', action='store_true', help='Output information for debugging etc.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--trajectories', type=float, default=7, help='length of summary - note this includes only the important states')
    parser.add_argument('--context', type=float, default=15, help='how many states to show around the chosen important state')
    parser.add_argument('--minimum-gap', type=float, default=50, help='how many states should we skip after showing the context for an important state.')


    args = parser.parse_args()
    
    # get current directory
    path = os.getcwd()
    if args.verbose:
        logger.info("Current Directory=%s", path)
        # prints parent directory
        print(os.path.abspath(os.path.join(path, os.pardir)))
    # use parent dir to save data, so we can keep the current folder small and portable
    directory = os.path.abspath(os.path.join(path, os.pardir))
    directory = os.path.join(directory, 'evaluate_agent_data')
    directory = os.path.join(directory, args.stream_folder)
    if args.verbose:
        logger.info("Now directory is: ")
        logger.info(directory)
    args.stream_folder = directory

    if args.generate_stream is True:
        stream_generator.generate_stream(args)
    if args.vis is True:
        if args.verbose:
            logger.info("Visualization turned on")


if __name__ == '__main__':
    main()
