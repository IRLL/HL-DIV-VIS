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
from datavault_callback import data_callback

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

def train(args, extra_args):
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
        
    env_type, env_id = get_env_type(args)
    #logger.info('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_highlights_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    #logger.info('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        load_path=args.load_path,
        save_path=args.save_path,
        **alg_kwargs
    )
    
    #print("Returned act: ")
    #print(model)

    return model, env


def build_env(args):
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        # TODO: Ensure willuse GPU when sent to SLURM (Add as a command-line argument)
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)


        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    if env_id == "MsPacmanNoFrameskip-v4":
        env = super_simple_dqn_wrapper.PacmanClearTheBoardRewardsWrapper(env)
        env = super_simple_dqn_wrapper.FearDeathWrapper(env)
    elif env_id == "FreewayNoFrameskip-v4":
        env = super_simple_dqn_wrapper.AltFreewayRewardsWrapper(env)
        env = super_simple_dqn_wrapper.FreewayUpRewarded(env)
        env.ale.setDifficulty(1)
    elif env_id == "JamesbondNoFrameskip-v4":
        env = super_simple_dqn_wrapper.FearDeathWrapper(env)
    return env


def get_env_type(args):
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

# BRITT ADDITION
def build_highlights_env(args):
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type = 'atari'
    env_id = args.env
    # Default alg is dqn, so make initial normal dqn environment
    env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
    
    #logger.info("About to check for training wrapper")
    # Now switch on the training-based args to add wrappers ass needed
    if args.training_wrapper == 'pacman_fear_only':
        env = super_simple_dqn_wrapper.fear_only(env)
        #logger.info("Training wrapper: " + str(args.training_wrapper))
    if args.training_wrapper == 'pacman_power_pill_only':
        env = super_simple_dqn_wrapper.pacman_power_pill_only(env)
        #logger.info("Training wrapper: " + str(args.training_wrapper))
    if args.training_wrapper == 'pacman_normal_pill_only':
        env = super_simple_dqn_wrapper.pacman_normal_pill_only(env)
    if args.training_wrapper == 'pacman_normal_pill_power_pill_only':
        env = super_simple_dqn_wrapper.pacman_normal_pill_power_pill_only(env)
    if args.training_wrapper == 'pacman_normal_pill_fear_only':
        env = super_simple_dqn_wrapper.pacman_normal_pill_fear_only(env)
    if args.training_wrapper == 'pacman_normal_pill_in_game':
        env = super_simple_dqn_wrapper.pacman_normal_pill_in_game(env)
    if args.training_wrapper == 'pacman_power_pill_fear_only':
        env = super_simple_dqn_wrapper.pacman_power_pill_fear_only(env)
    if args.training_wrapper == 'pacman_power_pill_in_game':
        env = super_simple_dqn_wrapper.pacman_power_pill_in_game(env)
    if args.training_wrapper == 'pacman_fear_in_game':
        env = super_simple_dqn_wrapper.pacman_fear_in_game(env)
    # training options for freeway (also specifies the environment)
    if args.training_wrapper == 'freeway_up_only':
        env = super_simple_dqn_wrapper.freeway_up_only(env)
    if args.training_wrapper == 'freeway_down_only':
        env = super_simple_dqn_wrapper.freeway_down_only(env)
    if args.training_wrapper == 'freeway_up_down':
        env = super_simple_dqn_wrapper.freeway_up_down(env)
    # training options for asterix (also specifies the environment)
    if args.training_wrapper == 'asterix_fear_only':
        env = super_simple_dqn_wrapper.fear_only(env)
    if args.training_wrapper == 'asterix_bonus_life_in_game':
        env = super_simple_dqn_wrapper.asterix_bonus_life_in_game(env)
    if args.training_wrapper == 'asterix_fear_in_game':
        env = super_simple_dqn_wrapper.asterix_fear_in_game(env)
    # training options for alien (also specifies the environment)
    if args.training_wrapper == 'alien_fear_only':
        env = super_simple_dqn_wrapper.fear_only(env)
    if args.training_wrapper == 'alien_pulsar_only':
        env = super_simple_dqn_wrapper.alien_pulsar_only(env)
    if args.training_wrapper == 'alien_eggs_only':
        env = super_simple_dqn_wrapper.alien_eggs_only(env)
    if args.training_wrapper == 'alien_eggs_pulsar_only':
        env = super_simple_dqn_wrapper.alien_eggs_pulsar_only(env)
    if args.training_wrapper == 'alien_eggs_fear_only':
        env = super_simple_dqn_wrapper.alien_eggs_fear_only(env)
    if args.training_wrapper == 'alien_eggs_in_game':
        env = super_simple_dqn_wrapper.alien_eggs_in_game(env)
    if args.training_wrapper == 'alien_pulsar_fear_only':
        env = super_simple_dqn_wrapper.alien_pulsar_fear_only(env)
    if args.training_wrapper == 'alien_pulsar_in_game':
        env = super_simple_dqn_wrapper.alien_pulsar_in_game(env)
    if args.training_wrapper == 'alien_fear_in_game':
        env = super_simple_dqn_wrapper.alien_fear_in_game(env)
    return env



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)

    # Use parser for specifying agent reward structure
    arg_parser = highlights_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
#        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])


    # Set up saving to a single, logical location on folder above the code base to avoid
    # Swelling the size of the code base with test outputs
    # get current date and time for default data output folder name
    if args.save_path is None:
        # datetime object containing current date and time
        now = datetime.now()
        #logger.info("now =" + str(now))

        # month_day_YY_H_M_S
        dt_string = now.strftime("%b%d_%Y_%H_%M_")
        dt_string = dt_string + str(args.training_wrapper)
        #logger.info("date and time =" + str(dt_string))
        args.save_path = dt_string
        # get current directory
        path = os.getcwd()
        # use parent dir to save data, so we can keep the current folder small and portable
        directory = os.path.abspath(os.path.join(path, os.pardir))
        directory = os.path.abspath(os.path.join(directory, os.pardir))
        directory = os.path.join(directory, 'train_agent_data')
        directory = os.path.join(directory, args.save_path)
        os.mkdir(directory)
        directory2 = os.path.join(directory, args.save_path)
        args.save_path = directory2
        #logger.info("Now save path is: ")
        #logger.info(args.save_path)
        args.log_path = os.path.join(directory, 'logs')
        os.mkdir(args.log_path)
        #logger.info("Now log path is: ")
        #logger.info(args.log_path)

    if args.save_path is not None and rank == 0:
        #logger.info("Inside custom run file and about to save model")
        save_path = osp.expanduser(args.save_path)
        args.log_path = os.path.join(directory, 'logs')
        
        
    model, env = train(args, extra_args)
    
    
    # Now save Model
    #logger.info("Model is: ")
    #logger.info(model)
    model.save(save_path)
    env.close()

    return model

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main(sys.argv)
