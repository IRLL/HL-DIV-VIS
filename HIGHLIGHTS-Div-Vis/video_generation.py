"""
    Utility script to generate all summarys for a given stream at once (1 HIGHLIGHTS-DIV summary and 10 random summaries).
"""

import coloredlogs, logging
import pandas as pd
import image_utils
import numpy as np
import sys
from highlights_state_selection import read_q_value_files, read_feature_files, compute_states_importance, highlights_div, random_state_selection, read_input_files
import os

#random seeds for the random summaries
seeds=[ 42, 1337, 1, 7, 13, 21, 153, 90,19234761, 291857957]



def print_df(stats_df):
    print("DF: ")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(stats_df)

def help_function(args, stream_folder, random_states_with_context, key_states):
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')

    logger.setLevel(logging.DEBUG)
    
    parameter_string = make_parameter_string(args)
    video_folder = os.path.join(stream_folder,'smooth_stream_vid_max/')
    
    if args.verbose:
        logger.info("Making highlights LRP video")
    image_folder = os.path.join(stream_folder, 'argmax_smooth/')
    video_name = 'highlights_div_lrp_' + parameter_string + '.mp4'
    image_utils.generate_video(args, image_folder, video_folder, video_name,
                               image_indices=key_states)
    
    if args.verbose:
        logger.info("Making highlights blur video")
    image_folder = os.path.join(stream_folder, 'blur_argmax/')
    video_name = 'highlights_div_blur_' + parameter_string + '.mp4'
    image_utils.generate_video(args, image_folder, video_folder, video_name,
                               image_indices=key_states)
    
    if args.verbose:
        logger.info("Making highlights video")
    image_folder = os.path.join(stream_folder, 'screen_smooth/')
    video_name = 'highlights_div_' + parameter_string + '.mp4'
    image_utils.generate_video(args, image_folder, video_folder, video_name,
                               image_indices=key_states)
                               
    counter = 0
    for random_state_set in random_states_with_context:
        if args.verbose:
            logger.info("Cycling through the pre-generated list of random states to make random movies.")
            logger.info("Starting with saliency maps")
        image_folder = os.path.join(stream_folder, 'argmax_smooth/')
        video_name = 'random_lrp_' + str(counter+1) + '_' + parameter_string + '.mp4'
        image_utils.generate_video(args, image_folder, video_folder, video_name,
                                   image_indices=random_state_set)
               
        if args.verbose:
            logger.info("And making normal video.")
        image_folder = os.path.join(stream_folder, 'screen_smooth/')
        video_name = 'random_' + str(counter+1) + '_' + parameter_string + '.mp4'
        image_utils.generate_video(args, image_folder, video_folder, video_name,
                                   image_indices=random_state_set)
        
        if args.verbose:
            logger.info("Now making random blurred video. Sounds less useful than it is...")
        image_folder = os.path.join(stream_folder, 'blur_argmax/')
        video_name = 'random_blur_' + str(counter+1) + '_' + parameter_string + '.mp4'
        image_utils.generate_video(args, image_folder, video_folder, video_name,
                                   image_indices=key_states)
        counter = counter + 1

    #

    #        for i in range(10):
    #            seed = seeds[i]
    #            random_states, random_states_with_args.context = random_state_selection(state_features_importance_df,
    #                                                                               args.trajectories,
    #                                                                               args.context, args.minimum_gap,seed=seed)
    #
    #            image_folder = os.path.join(stream_folder, 'argmax_smooth/')
    #            video_name = 'random_lrp_' + str(i+1) + '_' + parameter_string + '.mp4'
    #            image_utils.generate_video(args, image_folder, video_folder, video_name,
    #                                       image_indices=random_states_with_args.context)
    #
    #            image_folder = os.path.join(stream_folder, 'screen_smooth/')
    #            video_name = 'random_' + str(i+1) + '_' + parameter_string + '.mp4'
    #            image_utils.generate_video(args, image_folder, video_folder, video_name,
    #                                       image_indices=random_states_with_args.context)


def make_parameter_string(args):
    parameter_string = str(args.trajectories) + '_' + str(args.context) + '_' + str(args.minimum_gap)
    return parameter_string
 
 
def get_key_states_from_df(args, state_features_importance_df):
    if not (os.path.exists(args.stream_folder + '/summary_states_with_context.npy')):
        summary_states, summary_states_with_context = highlights_div(args, state_features_importance_df, args.budget, args.context, args.minimum_gap)
        np.save(args.stream_folder + '/summary_states.npy', summary_states)
        np.save(args.stream_folder + '/summary_states_with_context.npy', summary_states_with_context)
    else:
        print("Summary states file already exists")
        summary_states_with_context = np.load(args.stream_folder + '/summary_states_with_context.npy')
    return summary_states_with_context
    
    
def get_key_states(args, stream_folder, features='input', load_states=False):
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')

    logger.setLevel(logging.DEBUG)
    
    np.set_printoptions(threshold=sys.maxsize)
    if not (os.path.exists(stream_folder + '/summary_states_with_context.npy')):
        if args.verbose:
            logger.info("In Get key states and summary States file does not exist")
        parameter_string = make_parameter_string(args)
        

        q_values_df = read_q_value_files(stream_folder + '/q_values')
        if args.verbose:
            logger.info("Vid Gen and q_values_df is: ")
            print_df(q_values_df)
        states_q_values_df = compute_states_importance(q_values_df, compare_to='second')
        states_q_values_df.to_csv(stream_folder + '/states_importance_second.csv')
        states_q_values_df = pd.read_csv(stream_folder + '/states_importance_second.csv')
        if args.verbose:
            logger.info("Vid Gen and q_values_df is: ")
            print_df(q_values_df)
        if features == 'features':
            features_df = read_feature_files(stream_folder + '/features')
            features_df.to_csv(stream_folder + '/state_features.csv')
            features_df = pd.read_csv(stream_folder + '/state_features.csv')
            if args.verbose:
                logger.info("Vid Gen and q_values_df is: ")
                print_df(q_values_df)
            state_features_importance_df = pd.merge(states_q_values_df, features_df, on='state')
            state_features_importance_df = state_features_importance_df[['state', 'q_values', 'importance', 'features']]
            state_features_importance_df.to_csv(stream_folder + '/state_features_impoartance.csv')
            state_features_importance_df = pd.read_csv(stream_folder + '/state_features_impoartance.csv')

            state_features_importance_df['features'] = state_features_importance_df['features'].apply(lambda x:
                                                                                                      np.fromstring(
                                                                                                          x.replace('\n', '')
                                                                                                              .replace('[', '')
                                                                                                              .replace(']', '')
                                                                                                              .replace('  ',
                                                                                                                       ' '),
                                                                                                          sep=' '))
        elif features == 'input': #we need an extra case since the input arrays are too big to be saved in csv.
            if args.verbose:
                logger.info("Features set to input, creating state_features_importance_df")
            np.set_printoptions(threshold=sys.maxsize)
            features_df = read_input_files(stream_folder + '/state')
            
            features_df.to_csv(stream_folder + '/read_in_features.csv')
            state_features_importance_df = pd.merge(states_q_values_df, features_df, on='state')
            state_features_importance_df = state_features_importance_df[['state', 'q_values', 'importance', 'features']]
            state_features_importance_df.to_csv(stream_folder + '/state_features_impoartance.csv')
        else:
            logger.error('feature type not support.')
        
        if args.verbose:
            logger.info("About to call Highlights DIV algorithm from get_key_states")
        summary_states, summary_states_with_context = highlights_div(args, state_features_importance_df, args.trajectories, args.context, args.minimum_gap)
        np.save(stream_folder + '/summary_states.npy', summary_states)
        np.save(stream_folder + '/summary_states_with_context.npy', summary_states_with_context)
    else:
        print("In get key states and summary states file already exists")
        summary_states_with_context = np.load(stream_folder + '/summary_states_with_context.npy')
    return summary_states_with_context
    
def generate_videos(args, random_states_with_context):
    stream_folder = args.stream_folder
    image_indices = get_key_states(args, stream_folder, load_states = False)
    help_function(args, stream_folder, random_states_with_context, key_states = image_indices)

if __name__ == '__main__':
    generate_videos(args)
