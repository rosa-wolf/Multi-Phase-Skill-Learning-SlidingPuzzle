import os

import gymnasium as gym
import numpy as np
import torch
import time

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../../")
sys.path.append(mod_dir)

from FmReplayMemory import FmReplayMemory
from forwardmodel_simple_input.forward_model import ForwardModel

class SuccessCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, eval_freq: int,
                 seed: float,
                 size=[1, 2],
                 num_skills=2,
                 relabel=False,
                 logging=True,
                 eval_freq=500,
                 verbose=1):

        super().__init__(verbose)
        self.eval_freq = eval_freq

        # Todo: make a seperate evaluation environment
        # where we also load the current fm

        # Todo: we need access to current policy
        # Todo: we also need access to environment we train on

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        # save fm
        if self.verbose > 0:
            print("Saving initial model now")
        # don't save whole model, but only parameters
        torch.save(self.fm.model.state_dict(), self.save_path + "/fm")

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq == 0:
            # Todo: evaluate for 10 episodes

            # store if symbolic state changed
            pass



