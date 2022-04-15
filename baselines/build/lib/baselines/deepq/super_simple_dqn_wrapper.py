import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon = 0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon
    
    def action(self, action):
        return 0

class AltObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(AltObservationWrapper, self).__init__(env)
        print("Observations: ")
        print(self.observation_space)
        return None
    
    def observation(self, observation):
        return None

class AltRewardsWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(AltRewardsWrapper, self).__init__(env)
        return None
    
    def reward(self, reward):
        if reward > 10:
            print("Reward received: ")
            print(reward)
        return reward

class AltFreewayRewardsWrapper(gym.RewardWrapper):
    # chicken gets one point for getting across the highway
    # in normal game, set to a higher number to encourage
    # safely getting across the road
    def __init__(self, env):
        super(AltFreewayRewardsWrapper, self).__init__(env)
        return None
    
    def reward(self, reward):
        if reward > 0:
            print("Reward received: ")
            print(reward)
            reward = 10000
        return reward

class PacmanCherriesRewardsWrapper(gym.RewardWrapper):
    # chicken gets one point for getting across the highway
    # in normal game, set to a higher number to encourage
    # safely getting across the road
    def __init__(self, env):
        super(PacmanCherriesRewardsWrapper, self).__init__(env)
        return None
    
    def reward(self, reward):
        if reward == 100:
            print("Reward received: ")
            print(reward)
            reward = 10000
        else:
            reward = 0
        return reward

class FearDeathWrapper(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        # Add sharp negative reward to encourage fear
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = -1
        else:
            # since we have a seperate condition for fear with normal rewards,
            # set all other to 0 rewards
            reward = 0
        self.lives = lives
        return obs, reward, done, info
            
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs

class FreewayUpRewarded(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        if action == 1:
            reward = 1
        elif action == 2:
            reward = -0.5
        return obs, reward, done, info
            
            
#Starting here, custom wrappers correlate to chosen
#   training methods for use in evaluation by Highlights
#  algorithm for Britt's Thesis
    
class fear_only(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = - 10
        else:
            # if you didn't die, you lived, so hurray you get positive reinforcement
            reward = 1
        self.lives = lives
        return obs, reward, done, info
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs
        
class pacman_power_pill_only(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    # set anything besides power pill to 0 rewards
    def reward(self, reward):
        if reward < 50:
            reward = 0
        if reward > 99:
            reward = 0
        return reward
    
class pacman_normal_pill_only(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    # Test if getting 10 pts for eating normal pill
    # if so, keep pts, if no pill, negative points
    # if got pts for something out, no pts
    def reward(self, reward):
        if reward == 10:
            reward = 1
        elif reward == 0:
            reward = -1
        else:
            reward = 0
        return reward
    
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs


class pacman_normal_pill_power_pill_only(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    # Only keep points for eating pills
    # if no pills, negative reward (retreading old ground)
    # if pts for something else, no pts
    def reward(self, reward):
        if reward == 10:
            reward = 1
        else:
            if reward  == 0:
                reward = -1
            if (0 < reward < 50):
                reward = 0
            if reward > 99:
                reward = 0
        return reward

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs


class pacman_normal_pill_fear_only(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        # If a ghost just ate you, negative reward
        if lives < self.lives and lives > 0:
            # it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = -10
            # we want to encourage clearing the board so basic pills get a points boost
            if reward == 10:
                reward = 2
            # not eating a pill means retreading old ground
            # negative reinforcement
            elif reward == 0:
                reward = -1
            else:
                # if you didn't die, you lived, so hurray you get positive reinforcement
                reward = 1
            self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs
        
class pacman_normal_pill_in_game(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        # Since all other normal game rewards still apply,
        # set eating a normal pill to the highest reward possible
        if reward == 10:
            reward = 100
        # give negative reward for not eating a pill
        elif reward == 0:
            reward = -1
        # leave all other rewards as-is
        self.lives = lives
        return obs, reward, done, info
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs
    
class pacman_power_pill_fear_only(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = -100
        # set death by ghost to exact opposite of only other point source
        # which is eating a poser pill, 100 pts
        # All else gets 0 points
        else:
            if reward < 50:
                reward = 0
            if reward > 99:
                reward = 0
            self.lives = lives
        return obs, reward, done, info
        
        def reset(self, **kwargs):
            """Reset only when lives are exhausted.
            This way all states are still reachable even though lives are episodic,
            and the learner need not know about any of this behind-the-scenes.
            """
            if self.was_real_done:
                obs = self.env.reset(**kwargs)
            else:
                # no-op step to advance from terminal/lost life state
                obs, _, _, _ = self.env.step(0)
                self.lives = self.env.unwrapped.ale.lives()
            return obs
    
    
class pacman_power_pill_in_game(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        # Set eating a power pill to really high amount
        # leave all other rewards the same
        if reward > 50:
            if reward < 99:
                reward = 1000
        self.lives = lives
        return obs, reward, done, info
        
        def reset(self, **kwargs):
            """Reset only when lives are exhausted.
            This way all states are still reachable even though lives are episodic,
            and the learner need not know about any of this behind-the-scenes.
            """
            if self.was_real_done:
                obs = self.env.reset(**kwargs)
            else:
                # no-op step to advance from terminal/lost life state
                obs, _, _, _ = self.env.step(0)
                self.lives = self.env.unwrapped.ale.lives()
            return obs
    
class pacman_fear_in_game(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True
    
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = reward - 100
        return obs, reward, done, info
            
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs
    

class freeway_up_only(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # If action 1, means went up
        if action == 1:
            reward = 1
        return obs, reward, done, info


class freeway_down_only(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # If action 2, means went down
        if action == 2:
            reward = -1
        return obs, reward, done, info

class freeway_up_down(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # combine both positive and negative rewards
        if action == 1:
            reward = 1
        elif action == 2:
            reward = -1
        return obs, reward, done, info

# Can we get access to total reward here?
class asterix_bonus_life_in_game(gym.Wrapper):
    
    total_reward = 0
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # add to total reward
        self.total_reward = self.total_reward + reward
        #bonus life at 10,000, 30,000, 50,000, 80,000, 110,000 points,
        # and every 40,000 points thereafter.
        if self.total_reward == 10000:
            reward = reward+ 50
        elif self.total_reward == 30000:
            reward = reward+ 50
        elif self.total_reward == 50000:
            reward = reward+ 50
        elif self.total_reward == 80000:
            reward = reward+ 50
        elif self.total_reward == 110000:
            reward = reward+ 50
        elif self.total_reward - 110000 % 40000 == 0:
            reward = reward+ 50
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs

class asterix_fear_in_game(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = reward - 100
        elif reward == 0:
            # if you didn't die, you lived, so hurray you get positive reinforcement
            reward = reward + 1
        # leave all other rewards in place 
        self.lives = lives
        return obs, reward, done, info
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs

# Stopping edits here, will focus on training freeway and pacman for now
class alien_pulsar_only(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = -100
        else:
            if reward < 50:
                reward = 0
            if reward > 150:
                reward = 0
            self.lives = lives
        return obs, reward, done, info
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs

class alien_eggs_only(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def reward(self, reward):
        if reward == 10:
            reward = 1
        elif reward == 0:
            reward = -1
        else:
            reward = 0
        return reward
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs

class alien_eggs_pulsar_only(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def reward(self, reward):
        if reward == 10:
            reward = 1
        else:
            if reward  == 0:
                reward = -1
            if (1 < reward < 50):
                reward = 0
            if reward > 150:
                reward = 0
        return reward
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs


class alien_eggs_fear_only(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = - 100
        if reward == 10:
            reward = 10
        else:
            # if you didn't die, you lived, so hurray you get positive reinforcement
            reward = 1
            self.lives = lives
        return obs, reward, done, info
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs
        

class alien_eggs_in_game(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def reward(self, reward):
        if reward == 10:
            reward = 100
        elif reward == 0:
            reward = -1
        return reward
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs


class alien_pulsar_fear_only(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = -100
        else:
            if reward < 50:
                reward = 1
            if reward > 150:
                reward = 1
        self.lives = lives
        return obs, reward, done, info
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs


class alien_pulsar_in_game(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        
        #If you used a pulsar, get super high rewards. otherwise just normal
        if reward > 50:
            if reward < 150:
                reward = 2000
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs

class alien_fear_in_game(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
        # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            reward = reward - 100
        else:
            # if you didn't die, you lived, so hurray you get positive reinforcement
            reward = reward + 1
        self.lives = lives
        return obs, reward, done, info
        
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
        return obs
        

