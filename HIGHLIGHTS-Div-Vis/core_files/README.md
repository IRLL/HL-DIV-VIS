# Local and Global Explanations of Agent Behavior:Integrating Strategy Summaries with Saliency Map

This repository contains the implementation for the paper "Local and Global Explanations of Agent Behavior:Integrating Strategy Summaries with Saliency Map"(https://arxiv.org/abs/2005.08874).
This paper combines global explanations in the form of HIGHLIGHTS-DIV policy summaries (https://dl.acm.org/doi/10.5555/3237383.3237869) with LRP-argmax salieny maps (https://www.springerprofessional.de/enhancing-explainability-of-deep-reinforcement-learning-through-/17150184) 
by generating summaries of Atari agent behavior that is overlayd with saliency maps that show what information the agent used.

# Installation

We only tested with python 3.6.5. (conda install python==3.6.5 with Anaconda installed on Mac)
I am using Python 3.6.5 and Tensorflow 2.2.0 and keras-2.2.4 and pyyaml-5.3.1 and joblib-0.15.1 
I had to add tf.compat.v1 in place of all tf used in code to make this work.
It should be enough to install the given requirements.
For gym to work on a windows system you have to follow the instructions in *gym_for_windows.txt*.

*install_argmax.bat* is not neccessary anymore but can be used to update the argmax analyzer should the coresponding repository change.

# Summary Creation
The models in the folder *models* were trained with the openai-baselines repository https://github.com/openai/baselines.

Ideally, create a virtual environment. If using Anaconda, "conda create -n <yourenvironmentname> python=3.6.5 anaconda"

Then, activate using "conda activate highlightsENV"

*Tensorflow_to_Keras.py* converts the original tensorflow models to keras models.
Update all tf to tf.compat.v1.???
Then *stream_generator.py* creates a stream of gameplay, saving all states, visual frames, Q-values and raw LRP-argmax saliency maps (generated with *argmax_analyzer.py* from https://github.com/HuTobias/LRP_argmax). 
Here, you must install gym from openAI to get it working: pip3 install gym, I got version 0.17.2
And must install matplotlib: pip3 install matplotlib, I got version 3.2.2
And install cv2: pip3 install opencv-python, I got version 4.2.0.34
And install innvestigate: pip3 install innvestigate, I got version 1.0.8
And install theano: pip3 install theano, I got version 1.0.4 
And install skikit-image: pip3 install scikit-image, I got version 0.17.2

At the very end of *stream_generator.py*, *overlay_stream.py* is used to overlay each frame with a saliency map.
This can also be redone later using *overlay_stream.py* to save time while trying different overlay styles.

//not included in sream_generator
Based on those streams, *video_generation.py* generates the summary videos for the survey.
Hereby, *highlights_state_selection.py* is used to choose one set of states according to the HIGHLIGHTS-DIV algorithm and 10 different random sets of states for the random summaries.
The method that combines those frames to a video pipis implemented in *image_utils.py*.

# Subfolders
*Action_checks* and *Sanity_checks* check the action distribution of each agent and perform sanity checks for our saliency algorithm.

The videos we used in our survey are stored in the folder *Survey_videos* and the results of this survey are stored and evaluated in *Survey_results*. 
The *models* folder contains the trained agents we used and the streams we used are available upon request.

# Notes
https://github.com/szemenyeim/DynEnv is a well-documented environment, should follow this format in my thesis
https://github.com/davidcotton/gym-connect4 
https://awesomeopensource.com/project/maximecb/gym-minigrid
https://github.com/maximecb/gym-minigrid
