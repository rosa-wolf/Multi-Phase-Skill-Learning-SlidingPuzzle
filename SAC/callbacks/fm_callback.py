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

class FmCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, update_freq: int, save_path: str, seed: float, memory_size=500, sample_size=50, num_skills=2, verbose=1):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.sample_size = sample_size
        # initialize empty replay memory for fm
        self.buffer = FmReplayMemory(memory_size, seed)
        self.verbose = verbose
        self.num_skills = num_skills
        # initialize forward model (untrained)
        self.fm = ForwardModel(width=2,
                          height=1,
                          num_skills=self.num_skills,
                          batch_size=15,
                          learning_rate=0.001)

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
        """
        Update fm
        """
        if self.locals["dones"][0]:
            # Todo: save episode transition to buffer
            init_empty = self.fm.sym_state_to_input(self.locals['infos'][0]['init_sym_state'])
            out_empty = self.fm.sym_state_to_input(self.locals['infos'][0]['sym_state'])

            one_hot_skill = np.zeros(self.num_skills)
            one_hot_skill[self.locals['infos'][0]['skill']] = 1

            self.buffer.push(init_empty, one_hot_skill, out_empty)

        # only start training after buffer is filled a bit
        if self.n_calls % self.update_freq == 0:
            # Todo: update fm
            if len(self.buffer) >= self.sample_size:
                # update fm several times
                for _ in range(2):
                    # sample from buffer
                    episodes = self.buffer.sample(self.sample_size)

                    # reshape episodes for input to fm training
                    episodes = tuple(zip(*episodes))
                    episodes = np.array(episodes)
                    episodes = episodes.reshape((episodes.shape[0], 6))

                    train_loss, train_acc = self.fm.train(torch.from_numpy(episodes))

                    if self.verbose > 0:
                        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

                if self.verbose > 0:
                    print("Saving updated model now")
                # don't save whole model, but only parameters
                torch.save(self.fm.model.state_dict(), self.save_path + "/fm")
