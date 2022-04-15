"""
    Generates a stream of gameplay for a given agent.

    A folder 'stream' is created whose subfolders contain all the states, visually displayed frames, Q-values,
    saliency maps and features (output of the second to last layer).

    At the very end *overlay_stream* is used to overlay each frame with a saliency map.
    This can also be redone later using *overlay_stream* to save time while trying different overlay styles.
"""


import gym
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
import sys
import scipy
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import OrderedDict
import os
# from varname import nameof
import coloredlogs, logging

class DataVault:
    #dictionaries for storing all the info which will be shoved into dataframes later
    main_data_dict = OrderedDict()
    per_episode_action_distribution_dict = {}
    df_list = []
    args = []
    
    #keep track of steps
    step = 4

    def __init__(self):
        logger = logging.getLogger()
        coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
        logger.setLevel(logging.DEBUG)
        
    def print_df(self, stats_df):
        print("DF: ")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(stats_df)
            
    def df_to_csv(self, stream_directory):
        counter = 1
        for df in self.df_list:
            filename = "df" + str(counter) + ".csv"
            filepath = os.path.join(stream_directory, filename)
            df.to_csv(filepath, index=True)
            counter = counter + 1
        
    def store_data(self, action, action_name, action_episode_sums, action_total_sums, reward, done, info, lives, q_values, observation, mean_reward):
        logger = logging.getLogger()
        coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
        logger.setLevel(logging.DEBUG)
        #need to find some other way to store: Observations, argmax, features, q value
        
        end_of_episode = False
        end_of_epoch = False
        # If dataframe is not empty...
        if len(self.main_data_dict) != 0:
            #check contents of dictionary
#            logger.info("About to logger.info dictionary...")
#            for keys,values in self.main_data_dict.items():
#                logger.info(keys)
#                logger.info(values)
            lastElem = list(self.main_data_dict.keys())[-1]
#            logger.info("Last element is: " + str(lastElem))
            
            last_lives_left = self.main_data_dict[lastElem]['lives']
            episode = self.main_data_dict[lastElem]['episode']
            epoch = self.main_data_dict[lastElem]['epoch']
            episode_reward = self.main_data_dict[lastElem]['episode reward'] + reward
            epoch_reward = self.main_data_dict[lastElem]['epoch reward'] + reward
            total_reward = self.main_data_dict[lastElem]['total reward'] + reward
            self.step = self.step + 1
            episode = self.main_data_dict[lastElem]['episode']
            epoch = self.main_data_dict[lastElem]['epoch']
            episode_step = self.main_data_dict[lastElem]['episode step'] + 1
            epoch_step = self.main_data_dict[lastElem]['epoch step'] + 1
        else:
            last_lives_left = 3
            episode = epoch = episode_step = epoch_step = self.step
            episode_reward = epoch_reward = total_reward = reward
            
            
        eoe_flag = False
        #first check if new episode or new epoch started
        if (lives != last_lives_left):
            eoe_flag = True
            # Add to other dictionary, for stacked bar chart, as currently set
#            logger.info("Added to stacked chart dictionary")
            episode_reward = reward
            episode = episode + 1
            episode_step = 1
            end_of_episode = True
            for x in range(len(action_episode_sums)):
                action_episode_sums[x] = 0
            
            if (done):
            # Have used up all three lives, therfore an "epoch" is over, and need to zero out accumulators
                epoch_reward = reward
                epoch = epoch + 1
                end_of_epoch = True
                epoch_step = 1
#                logger.info("end of episode and epoch is true ")
                eoe_flag = True
        
        # Up correct action sum
        temp_action_episode_sum = action_episode_sums[action]
        action_episode_sums[action] = temp_action_episode_sum + 1
        
        temp_action_total_sum = action_total_sums[action]
        action_total_sums[action] = temp_action_total_sum + 1
#        logger.info("end of episode and epoch is: ")
#        logger.info(end_of_episode)
#        logger.info(end_of_epoch)
        step_stats = { self.step: {
            'action_name': action_name,
            'action': action,
            'reward': reward,
            'episode reward': episode_reward,
            'epoch reward': epoch_reward,
            'total reward': total_reward,
            'lives': lives,
            'end of episode': end_of_episode,
            'end of epoch': end_of_epoch,
            'info': info,
            'episode': episode,
            'episode step': episode_step,
            'epoch': epoch,
            'epoch step': epoch_step,
            'state': self.step,
            'q_values': np.squeeze(q_values),
            'observation': np.squeeze(observation),
            'mean reward': mean_reward
            }
        }
            
            
        # add carefully the action sums to the dictionary
        for action_number in range(len(action_episode_sums)):
#            logger.info("Action in list: ")
#            logger.info(action_number)
            index_name = "action " + str(action_number) + " episode sum"
            step_stats[self.step][index_name] = action_episode_sums[action_number]
            index_name = "action " + str(action_number) + " total sum"
            step_stats[self.step][index_name] = action_total_sums[action_number]
            
#            logger.info("About to logger.info dictionary after steps update...")
#            for keys,values in self.main_data_dict.items():
#                logger.info(keys)
#                logger.info(values)
            
                
    #    logger.info(step_stats)
        #add to the dictionary
        self.main_data_dict.update(step_stats)
#        logger.info("Updated main dictionary: ")
#        logger.info(self.main_data_dict)
        
        return (action_episode_sums, action_total_sums)

    def make_dataframes(self):
        np.set_printoptions(threshold=sys.maxsize)
        main_df = pd.DataFrame.from_dict(self.main_data_dict, orient='index')
        stacked_bar_df = pd.DataFrame.from_dict(self.per_episode_action_distribution_dict, orient='index')
        self.df_list.append(main_df)
        self.df_list.append(stacked_bar_df)
        
#        q_values_df = main_df[['state','q_values']].copy()
##        q_values_df.step.name = 'state'
#        print("Have made the following q_value df: ")
#        self.print_df(q_values_df)
#        self.df_list.append(q_values_df)
#        
#        features_df = main_df[['state','features']].copy()
##        features_df.step.name = 'state'
#        print("Have made the following features df: ")
#        self.print_df(features_df)
#        self.df_list.append(features_df)
#        
##        raw_screen_features_df = main_df[['state','raw_screen_data']].copy()
##        #rename the raw_screen_data to features
##        raw_screen_features_df.rename(columns = {'raw_screen_data':'features'}, inplace = True)
###        raw_screen_features_df.step.name = 'state'
#        raw_screen_features_df = read_input_files(args.stream_folder + '/state')
#        print("Have made the following features df: ")
#        self.print_df(raw_screen_features_df)
#        self.df_list.append(raw_screen_features_df)
        
#        state_features_importance_df = self.get_state_importance(q_values_df, raw_screen_features_df)
#        state_features_importance_df.index.name = 'state'
#        self.df_list.append(state_features_importance_df)

    def get_state_importance(self, q_values_df, features_df):
        states_q_values_df = compute_states_importance(q_values_df, compare_to='second')
        state_features_importance_df = pd.merge(states_q_values_df, features_df, on='state')
        # make sure the index is the state
#        state_features_importance_df.index.name = 'state'
        
        # Using the features=='input' from video_generation.py
        state_features_importance_df = state_features_importance_df[['state','q_values', 'importance', 'features']]
        
        print("state_features_importance_df")
        self.print_df(state_features_importance_df)
        return state_features_importance_df
