import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.keras import models
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env, highlights_arg_parser
from baselines.common.tf_util import get_session
from baselines.common.data_storage import DataVault
from baselines import logger
from importlib import import_module
from deepq import super_simple_dqn_wrapper
from datetime import datetime
import os
import logging
import coloredlogs



# Callback created by following this tutorial: https://www.youtube.com/watch?reload=9&v=Z-yVErKvZrg
# Should be called on each step
def data_callback(lcl, _glb):
    print("All possible local values: ")
    for item in lcl:
        print(item)
        print(lcl[item])
    print("Other values...................................................................... ")
    for item in _glb:
        print(item)
        print(_glb[item])
        
    # First, check if it is time to checkpoint the model so we don't
    # have really bad fallout from a crash at the end
#    if lcl['t'] % lcl['checkpoint_frequency'] == 0:
        
    
    
    
    
    # Next see if it is time to (less frequent) save the model
#    if lcl['t'] % lcl['save_frequency'] == 0:
    
    
    
    
    # Finally, check if we are done, and if so, convert the
    # datavault object to dataframes and save as CSVs in
    # the designated folder
#    if lcl['t'] == lcl['total_timesteps']
#    lcl['dv'].make_dataframes(
    
        
        
        
    # If return true, halts training
    # If returns false, training continues
    return True
    
