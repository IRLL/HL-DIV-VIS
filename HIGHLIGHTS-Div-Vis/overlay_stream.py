"""
    Implements the function overlay_stream(args.stream_folder), which overlays all frames in the given directory with the
    saliency maps stored in the given directory.
"""

import coloredlogs, logging
import os
import sys
import image_utils
import pandas as pd
import cv2
import numpy as np
import stream_generator
import video_generation as video_generation
import tensorflow as tf
from highlights_state_selection import read_q_value_files, read_feature_files, compute_states_importance, highlights_div, random_state_selection, read_input_files
from video_generation import get_key_states_from_df, get_key_states

#random seeds for the random summaries
seeds=[ 42, 1337, 1, 7, 13, 21, 153, 90,19234761, 291857957]

def interpolate(array1, array2, t):
    '''
    linear interpolation between two frames of a state
    :param array1: starting array
    :param array2: end array
    :param t: time parameter, goes from -1 to 3 ( 0=0.25, 3=1 in the normal interpolation formula)
    :return: the interpolated array
    '''
    t = (t * 0.25) + 0.25
    return (array2 * t) + (array1 * (1 - t))
    
def get_random_states_list(args, logger, key_states_with_context):
    ''' Get a list of all the states we need saliency overlays for,
        to save on computation of overlaying everything '''
        
    state_features_importance_df = pd.read_csv(args.stream_folder + '/state_features_impoartance.csv')
    
    consolidated_random_states_list_without_repeats = []
    random_states = []
    random_states_with_context = []
    for i in range(10):
        seed = seeds[i]
        random_states_item, random_states_with_context_item = random_state_selection(state_features_importance_df, args.trajectories, args.context, args.minimum_gap,seed=seed)
        if args.verbose:
            logger.debug("Generated random state: ")
            logger.debug(random_states_item)
            logger.debug("With context: ")
            logger.debug(random_states_with_context_item)
        random_states.append(random_states_item)
        random_states_with_context.append(random_states_with_context_item)
        for state_number in random_states_with_context[i]:
            if state_number not in consolidated_random_states_list_without_repeats:
                consolidated_random_states_list_without_repeats.append(state_number)
    
    if args.verbose:
        logger.debug("Randome states:")
        logger.debug(random_states)
        logger.debug("With context:")
        logger.debug(random_states_with_context)
        logger.debug("Consolidated list: ")
        logger.debug(consolidated_random_states_list_without_repeats)
    return random_states, random_states_with_context, consolidated_random_states_list_without_repeats

def overlay_stream(args):
    '''
    overlays all screens in the args.stream_folder
    :param args.stream_folder: see above
    :return: nothing
    '''
    
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')

    logger.setLevel(logging.DEBUG)
    
    stream_folder = args.stream_folder
    image_folder = stream_folder + "/screen"
    raw_argmax_base = stream_folder + "/raw_argmax/raw_argmax"
    save_folder = stream_folder + "/argmax_smooth"
    save_folder2 = stream_folder + "/screen_smooth"
    save_folder3 = stream_folder + "/blur_argmax"
    if not (os.path.isdir(save_folder2)):
        os.makedirs(save_folder2)
    if not (os.path.isdir(save_folder3)):
        os.makedirs(save_folder3)

    images = [img for img in os.listdir(image_folder)]
    images = image_utils.natural_sort(images)
    
    print("Traditionally generated key states: ")
    key_states_with_context = get_key_states(args, stream_folder, features='input', load_states=False)
    
    np.set_printoptions(threshold=sys.maxsize)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Traditionally generated key states: ")
        print(key_states_with_context)
        
    # Get only the important states
#    print("Key states from df: ")
#    key_states_with_context=get_key_states_from_df(args, state_features_importance_df)
#
#
#    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#        print("Key states from df: ")
#        print(key_states_with_context)
    
    
    # Then generate needed number of random states
    random_states, random_states_with_context, consolidated_random_states_list_without_repeats = get_random_states_list(args, logger, key_states_with_context)

    if args.verbose:
        logger.debug("make original saliency maps for all necessary states")
    old_saliency_map = None
    old_image = None
    for image in images:
        try:
            image_str = image.split('_')
            state_index = int(image_str[1])
            frame_index = int(image_str[2].replace(".png", ""))
            
            if args.verbose:
                logger.info("About to compare state to key states list....")
                        
            #            image_indices = [4,5,6,7]
            if (state_index in consolidated_random_states_list_without_repeats) or (consolidated_random_states_list_without_repeats is None):
                if args.verbose:
                    logger.info("State " + str(state_index) + " is in key states list: " + str(consolidated_random_states_list_without_repeats))
                i = cv2.imread(os.path.join(image_folder, image))
                i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
                if old_image is not None:
                    smooth_i = np.maximum(old_image,i)
                    old_image = i
                    i = smooth_i
                else:
                    old_image = i

                image_utils.save_image(os.path.join(save_folder2, image) ,i)

                saliency_filename = raw_argmax_base + "_" + str(state_index) + ".npy"
                saliency_map = np.load(saliency_filename)
                saliency_map = image_utils.normalise_image(saliency_map)
                if saliency_map.sum() > 0.9 * saliency_map.shape[0] * saliency_map.shape[1] * saliency_map.shape[2]:
                    if args.verbose:
                        logger.info(state_index)
                    saliency_map = np.zeros(saliency_map.shape)
                if old_saliency_map is not None:
                    saliency_map = interpolate(old_saliency_map, saliency_map, frame_index)
                saliency = image_utils.output_saliency_map(saliency_map[:, :, 3], i, edges=False)
                index = str(state_index) + '_' + str(frame_index)
                stream_generator.save_frame(saliency, save_folder + "/argmax", index)
                if frame_index == 3:
                    old_saliency_map = saliency_map
        except Exception as e:
            logger.error(e)
            logger.error('Try next image.')
            continue
            
    
    if args.verbose:
        logger.debug("make blur-based saliency maps for all necessary states")
    old_saliency_map = None
    old_image = None
    for image in images:
        try:
            image_str = image.split('_')
            state_index = int(image_str[1])
            frame_index = int(image_str[2].replace(".png", ""))
            
            if args.verbose:
                logger.info("About to compare state to key states list....")
            
#            image_indices = [4,5,6,7]
            if (state_index in consolidated_random_states_list_without_repeats) or (consolidated_random_states_list_without_repeats is None):
                if args.verbose:
                    logger.info("State " + str(state_index) + " is in key states list: " + str(consolidated_random_states_list_without_repeats))
                i = cv2.imread(os.path.join(image_folder, image))
                i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
                if old_image is not None:
                    smooth_i = np.maximum(old_image,i)
                    old_image = i
                    i = smooth_i
                else:
                    old_image = i

    #            image_utils.save_image(os.path.join(save_folder, image) ,i)

                saliency_filename = raw_argmax_base + "_" + str(state_index) + ".npy"
                saliency_map = np.load(saliency_filename)
                saliency_map = image_utils.normalise_image(saliency_map)
                if saliency_map.sum() > 0.9 * saliency_map.shape[0] * saliency_map.shape[1] * saliency_map.shape[2]:
                    if args.verbose:
                        logger.info("state index is: " + str(state_index))
                    saliency_map = np.zeros(saliency_map.shape)
                if old_saliency_map is not None:
                    saliency_map = interpolate(old_saliency_map, saliency_map, frame_index)
                saliency = image_utils.output_blur_saliency_map(saliency_map[:, :, 3], i, edges=False)
                index = str(state_index) + '_' + str(frame_index)
                stream_generator.save_frame(saliency, save_folder3 + "/argmax", index)
                if frame_index == 3:
                    old_saliency_map = saliency_map
        except Exception as e:
            logger.error(e)
            logger.error('Try next image.')
            continue
            
    if args.generate_video is True:
        video_generation.generate_videos(args, random_states_with_context)


if __name__ == "__main__":

    #get rid of distracting TF errors
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    stream_folder = 'stream'
    overlay_stream(stream_folder, args)
