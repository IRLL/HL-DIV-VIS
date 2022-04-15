"""
    Generates a stream of gameplay for a given agent.

    A folder 'stream' is created whose subfolders contain all the states, visually displayed frames, Q-values,
    saliency maps and features (output of the second to last layer).

    At the very end *overlay_stream* is used to overlay each frame with a saliency map.
    This can also be redone later using *overlay_stream* to save time while trying different overlay styles.
"""


import gym
import matplotlib.pyplot as plt
from custom_atari_wrapper import atari_wrapper, AltRewardsWrapper
from highlights_state_selection import read_q_value_files, read_feature_files, compute_states_importance, highlights_div, random_state_selection, read_input_files
#from RAM_analysis import RAM_Vault
import numpy as np
import keras
from argmax_analyzer import Argmax
import overlay_stream
from data_storage import DataVault
#from gym_minigrid.wrappers import *
#import gym_maze
import pandas as pd
import sys
import scipy
import seaborn as sns
import h5py
import coloredlogs, logging

#Quickfix for argmax
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def vis_testing(stats_df, directory):
    column_array = stats_df.dtypes
    logger.info(column_array)
    stats_df.describe()

    sns.scatterplot(x='total reward',y='episode',data=stats_df)
    save_file = os.path.join(directory, 'plot1.png')
    plt.savefig(save_file, dpi = 200)
    
    sns.scatterplot(x='total reward',y='step',data=stats_df)
    save_file = os.path.join(directory, 'plot2.png')
    plt.savefig(save_file, dpi = 200)

    sns.relplot(x='step',y='total reward',hue='lives',size='reward',data=stats_df)
    save_file = os.path.join(directory, 'plot3.png')
    plt.savefig(save_file, dpi = 200)

    g = sns.relplot(x="action_name", y="reward", kind="line", data=stats_df)
    g.fig.autofmt_xdate()
    save_file = os.path.join(directory, 'plot4.png')
    plt.savefig(save_file, dpi = 200)

    age_vs_hours_per_week = sns.relplot(x="action_name", y="total reward", kind="line", data=stats_df)
    save_file = os.path.join(directory, 'plot5.png')
    plt.savefig(save_file, dpi = 200)

    age_vs_hours_per_week = sns.relplot(x="episode", y="total reward", kind="line",sort=False, data=stats_df)
    save_file = os.path.join(directory, 'plot6.png')
    plt.savefig(save_file, dpi = 200)

    sns.relplot(x='step',y='action',hue='reward',size='episode',col='total reward',data=stats_df)
    save_file = os.path.join(directory, 'ploy7.png')
    plt.savefig(save_file, dpi = 200)

    sns.relplot(x='total reward',y='reward',hue='step',size='lives',col='epoch',data=stats_df)
    plt.savefig('plot2.png', dpi = 200)

    sns.relplot(x='action 0 episode_sum',y='action 1 episode_sum',hue='action 2 episode_sum',size='action 3 episode_sum',col='action 4 episode_sum',row='action 5 episode_sum',height=5,data=stats_df)
    plt.savefig('plot3.png', dpi = 200)

    sns.catplot(x="step",y="total reward",kind='bar',hue='action',data=stats_df)
    plt.savefig('plot4.png', dpi = 200)

    ax = sns.catplot(x='action',kind='count',data=stats_df,orient="h")
    ax.fig.autofmt_xdate()
    plt.savefig('plot5.png', dpi = 200)

    ax = sns.catplot(x='episode',y='reward',hue='step',kind='point',data=stats_df)
    ax.fig.autofmt_xdate()
    plt.savefig('plot6.png', dpi = 200)

    sns.catplot(x="total reward", y="step", hue="action_name",
            col="action", aspect=.6,
            kind="box", data=stats_df);
    plt.savefig('plot7.png', dpi = 200)
#
##        Categorical scatterplots:
##    stripplot() (with kind=”strip”; the default)
##    swarmplot() (with kind=”swarm”)
##    Categorical distribution plots:
##    boxplot() (with kind=”box”)
##    violinplot() (with kind=”violin”)
##    boxenplot() (with kind=”boxen”)
##    Categorical estimate plots:
##    pointplot() (with kind=”point”)
##    barplot() (with kind=”bar”)
##    countplot() (with kind=”count”)
#    sns.catplot(x="step",y="total reward",data=stats_df)
#    plt.savefig('plot8.png', dpi = 200)
#
#    #sns.catplot(x="age",y="relationship",kind='swarm',data=census_data)
#    #  or
#    #sns.swarmplot(x="relationship",y="age",data=census_data)
#    sns.catplot(x="step", y="total reward", kind="swarm", data=stats_df);
#    plt.savefig('plot9.png', dpi = 200)
#
#    sns.catplot(x="step", y="total reward", hue="action", kind="swarm", data=stats_df);
#    plt.savefig('plot10.png', dpi = 200)
#
#    sns.catplot(x="step",y="total reward",kind='box',data=stats_df)
#    plt.savefig('plot11.png', dpi = 200)
#
#    sns.catplot(x="step",y="total reward",kind='box',hue='episode',data=stats_df)
#    plt.savefig('plot12.png', dpi = 200)
#
#    sns.catplot(x="step",y="total reward",kind='violin',data=stats_df)
#    plt.savefig('plot13.png', dpi = 200)
#
#    sns.catplot(x="step",y="total reward",kind='violin',bw=.15, cut=0,data=stats_df)
#    plt.savefig('plot14.png', dpi = 200)
#
#    sns.catplot(x="step",y="total reward",kind='bar',data=stats_df)
#    plt.savefig('plot15.png', dpi = 200)

def save_frame(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
        os.rmdir(save_file)
    plt.imsave(save_file + '_' + str(frame) + '.png', array)

def save_array(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
        os.rmdir(save_file)
    np.save(save_file + '_' + str(frame) + '.npy', array)

def save_q_values(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
        os.rmdir(save_file)
    save_file = save_file + '_' + str(frame) + '.txt'
    with open(save_file, "w") as text_file:
        text_file.write(str(array))

def save_raw_data(array,save_file, frame):
    '''
    saves a raw state or saliency map as array and as image
    :param array: array to be saved
    :param save_file: file path were the data should be saved
    :param frame: the frame index of the file
    :return: None
    '''
    save_array(array,save_file, frame)
    image = np.squeeze(array)
    image = np.hstack((image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]))
    save_frame(image, save_file, frame)
    
def save_raw_data_with_blur(array,save_file, frame):
    '''
    saves a raw state or saliency map as array and as image
    :param array: array to be saved
    :param save_file: file path were the data should be saved
    :param frame: the frame index of the file
    :return: None
    '''
    blur_array = simple_non_uniform_blur(frame, array)
    save_array(array,save_file, frame)
    image = np.squeeze(blur_array)
    image = np.hstack((image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]))
    save_frame(image, save_file, frame)

def get_feature_vector(model, input):
    '''
    returns the output of the second to last layer, which act similar to a feature vector for the DQN-network
    :param model: the model used for prediction
    :param input: the input for the prediction
    :return:
    '''
    helper_func = keras.backend.function([model.layers[0].input],
                                  [model.layers[-2].output])
    features = helper_func([input])[0]
    features = np.squeeze(features)
    return features
    
def generate_stream(args):
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')

    logger.setLevel(logging.DEBUG)

    #create an object of class Data Vault and of RAM Vault
    dv = DataVault()
#    rv = RAM_Vault()
    step = 1

    #use a different start to get states outside of the highlights stream
    fixed_start = False

    np.random.seed(42)
    
    #pull model from passed in arguments
    model_path = os.path.join('models', args.agent_model)
    if args.verbose:
        logger.info("Now model path is: ")
        logger.info(model_path)
    
    model = keras.models.load_model(model_path)
    if args.verbose:
        logger.info("Generating stream with model: ")
        logger.info(model)
    
    # Set up variable for data tracking
    last_lives_left = 3
    action_names = []
    action_episode_sums = []
    action_total_sums = []
    
    steps = args.num_steps

    if args.verbose:
        logger.info("Model Summary....................................")
        logger.info(model.summary())

    analyzer_arg = Argmax(model)

    mean_reward = 0
    total_reward = 0
    step = 1
    episode = 1
    epoch = 1
    reward_list = []
    
    env = gym.make(args.gym_env)
#        logger.info("HERE and now environment is: ")
#        logger.info(args.gym_env)
    action_names = env.unwrapped.get_action_meanings()
    if args.verbose:
        logger.info(action_names)
    for x in range(len(action_names)):
        action_episode_sums.append(0)
        action_total_sums.append(0)
        if args.verbose:
            logger.info("Action sums: ")
            logger.info(action_episode_sums)
    
    env = AltRewardsWrapper(env)
    env.reset()
    wrapper = atari_wrapper(env)
    wrapper.reset(noop_max=1)
    if fixed_start :
        rapper.fixed_reset(300,2) #used  action 3 and 4

    directory = ''
    directory = os.path.join(directory, args.stream_folder)
    save_file_argmax = os.path.join(directory, 'argmax', 'argmax')
    save_file_argmax_raw = os.path.join(directory, 'raw_argmax', 'raw_argmax')
    save_file_screen = os.path.join(directory, 'screen', 'screen')
    save_file_state = os.path.join(directory, 'state', 'state')
    save_file_q_values = os.path.join(directory, 'q_values', 'q_values')
    save_file_features = os.path.join(directory, 'features', 'features')
    scores_file = os.path.join(directory, 'scores.txt')
    if args.verbose:
        logger.info("Made scores file")
    df_file = os.path.join(directory, 'df_stats.csv')
    if args.verbose:
        logger.info("DF file is: ")
        logger.info(df_file)
    average_score_file = os.path.join(directory, 'average_score.txt')

    for _ in range(steps):
        if _ < 4:
            action = env.action_space.sample()
            if args.verbose:
                logger.info("Now taking action " + str(action))
            # to have more controll over the fixed starts
            if fixed_start:
                action=0
            output = [0]
            features = [0]
            argmax = 0
            features = [0]
            my_input = [0]
        else:
            my_input = np.expand_dims(stacked_frames, axis=0)
#            logger.info("first input: ")
#            logger.info(my_input)
#            logger.info("squeezed input")
#            logger.debug(np.squeeze(my_input))
            
            if args.verbose:
                logger.info("MY_INPUT is: " + str(my_input))
            output = model.predict(my_input, verbose = 0)  #this output corresponds with the output in baseline if --dueling=False is correctly set for baselines.
            # save model predictions
            save_q_values(output, save_file_q_values, _)
            features = get_feature_vector(model, my_input)
            save_q_values(features,save_file_features,_)
            save_array(features, save_file_features,_)

            action = np.argmax(np.squeeze(output)[0])
            
            #action = env.action_space.sample()
            if args.verbose:
                logger.info("Now taking action " + str(action))

            #analyzing
            argmax = analyzer_arg.analyze(my_input)
            argmax = np.squeeze(argmax)
            # save raw saliency
            save_raw_data(argmax, save_file_argmax_raw, _)

            #save the state
            save_raw_data(my_input,save_file_state, _)

            #save screen output, and screen + saliency
            for i in range(len(observations)):
                index = str(_) + '_' + str(i)
                observation = observations[i]
                if args.verbose:
                    logger.info("Obs is: ")
                    logger.info(observation)
                save_frame(observation, save_file_screen, index)

    
        stacked_frames, observations, reward, done, info = wrapper.step(action)
        total_reward = total_reward + reward
        mean_reward = (sum(reward_list) + total_reward)/step
        if (args.verbose):
            logger.info("Step " + str(step) + " out of " + str(args.num_steps))
        step = step + 1
        ram = env.unwrapped._get_ram()
        if args.verbose:
            logger.info("Action is: ")
            logger.info(action_names[action])
            logger.info("RAM is: ")
            logger.info(ram)
#        rv.store_ram_info(action, ram)
        # only collect data after the first four steps
        if _ >= 4:
            # Add call here to update Pandas dataframe and output info for analysis
            lives = env.ale.lives()
            action_episode_sums, action_total_sums = dv.store_data(action, action_names[action], action_episode_sums, action_total_sums, reward, done, info, lives, output, argmax, observations, mean_reward)
    
        if done:
            if args.verbose:
                logger.info('total_reward',total_reward)
            reward_list.append(total_reward)
            
            total_reward = 0
            
        if (args.watch_agent):
            env.render()

    reward_list.append(total_reward)
    average_reward = np.mean(reward_list)
    with open(scores_file, "w") as text_file:
        text_file.write(str(reward_list))
    with open(average_score_file, "w") as text_file:
        text_file.write(str(average_reward))

    import datetime
    if args.verbose:
        logger.info('Time:')
        logger.info(datetime.datetime.now())
    
    #produce some visualizations with the data
    #vis_testing(stats_df, directory)
    
    #before processing images because that's slow
    dv.make_dataframes(args)
#    dv.df_to_hdf5(directory)
    dv.df_to_csv(directory)
    
    #overlays the stream of frames with the saliency maps.
    overlay_stream.overlay_stream(args)
    return()

if __name__ == '__main__':
    generate_stream(args)
