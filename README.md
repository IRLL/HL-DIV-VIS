# HL-DIV-VIS
# How AI Plays Games: The Code

This repository contains the implementation for the HIGHLIGHTS-Vis algorithm by Britt Davis Pierson, based on work from Huber, Amir and Amir, and De la Cruz. There are three main folders. Baselines containes code to train RL agents using a customized version of OpenAI baselines' DQN code in the Ms. Pacman environment. The HIGHLIGHTS-Div-Vis folder contains code to evaluate agents after they are trained. The evaluation has been altered with a Data Scraper, which will collect observational data from the agent during the evaluation, and save it to file.  

# Installation

The project was built with Python 3.6.5 and Tensorflow 2.2.0 and keras-2.2.4 and pyyaml-5.3.1 and joblib-0.15.1 
From the main folder, the required packages can be installed to a new virtual environment with the following commands:
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Agent Creation


# Summary Creation
The models in the folder *HIGHLIGHTS-Div-Vis>models* were trained with the fork of the openai-baselines repository in the baselines folder

In the folder, *HIGHLIGHTS-Div-Vis>model_translation_preprocessing*, *Tensorflow_to_Keras.py* converts the original tensorflow models to keras models.

To evaluate the Standard_9M agent for 10,000 time steps, navigate to the HIGHLIGHTS-Div-Vis folder in terminal and enter the following command:
```
python run_model.py --agent-model standard_9M.h5 --generate-stream --overlay-saliency --num-steps 10000 --watch-agent --verbose
```

The file *run_model.py* handles which parts of the evaluation to run, and takes the following arguments: 
```
optional arguments:
  -h, --help            show this help message and exit
  --gym-env GYM_ENV     OpenAi Gym environment ID
  --agent-model AGENT_MODEL
                        Name of model saved in model folder which you want to
                        use, including h5 extension
  --cuda-devices CUDA_DEVICES
  --gpu-fraction GPU_FRACTION
  --cpu-only
  --convert-model       run the algorithm to convert a tensorflow model to
                        keras model
  --generate-stream     output a stream into a new folder
  --overlay-saliency    use to overlay a saliency map onto the screenshots
  --stream-folder STREAM_FOLDER
  --generate-video      creates video of the summary states and screenshots
                        with saliency maps
  --minigrid            Use the minigrid environment
  --num-steps NUM_STEPS
  --watch-agent         shows a window with the agent acting in real-time
  --vis                 generate additional plots and charts
  --verbose             Output information for debugging etc.
  --trajectories TRAJECTORIES
                        length of summary - note this includes only the
                        important states
  --context CONTEXT     how many states to show around the chosen important
                        state
  --minimum-gap MINIMUM_GAP
                        how many states should we skip after showing the
                        context for an important state.
```

The first time you run an evalaution, a fourth folder will be created at the top level, called *evaluate_agent_data*. The results from the evaluation will be saved inside this folder, as a timestamped sub-folder. As long as you wait a minute between evaluations, a new folder will be created for a new evaluation. 

# Making Visuals
Once you have finished training an agent and evaluating it, you can use the data collected during evaluation to make charts. To do so, navigate in terminal to the *Notebooks>Visuals* folder, and start a Jupyter Notebook session by typing, `Jupyter Notebook` and hitting enter. Open the notebook titled *VisualsToExploreTheEvaluationData.ipynb*, and redirect the appropriate variables to where the evaluation data was saved.
