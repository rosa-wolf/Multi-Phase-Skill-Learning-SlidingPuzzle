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

    def __init__(self, update_freq: int, save_path: str,
                 seed: float, memory_size=500,
                 sample_size=30,
                 size=[1, 2],
                 num_skills=2,
                 relabel=False,
                 verbose=1):

        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.sample_size = sample_size
        self.relabel = relabel
        self.relabel_buffer = {"max_reward": [],
                                "max_skill": []}

        # initialize empty replay memory for fm
        self.buffer = FmReplayMemory(memory_size, seed)
        self.verbose = verbose
        self.num_skills = num_skills
        # initialize forward model (untrained)
        self.fm = ForwardModel(width=size[1],
                          height=size[0],
                          num_skills=self.num_skills,
                          batch_size=sample_size,
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

        if self.locals["dones"] != 0:
            print("append infos to buffer")
            print("infos = ", self.locals["infos"][0])
            self.relabel_buffer["max_reward"].append((self.locals["infos"][0])["max_reward"])
            self.relabel_buffer["max_skill"].append((self.locals["infos"][0])["max_skill"])
            print("buffer = ", self.relabel_buffer)

        # only start training after buffer is filled a bit
        if self.n_calls % self.update_freq == 0:
            # Todo: update fm
            if len(self.buffer) >= self.sample_size:
                # update fm several times
                for _ in range(2):

                    # put sampling batch(es) from buffer into forward model train function
                    train_loss, train_acc = self.fm.train(self.buffer)

                    if self.verbose > 0:
                        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

                if self.verbose > 0:
                    print("Saving updated model now")
                # don't save whole model, but only parameters
                torch.save(self.fm.model.state_dict(), self.save_path + "/fm")


    def _on_rollout_end(self) -> bool:
        """
        Update fm
        """
        print("---------------------------------")

        print("num_collected_episodes = ", self.locals["num_collected_episodes"])
        print("replay_buffer_done = ", np.where((self.locals["replay_buffer"]).dones == 1))

        # number of episodes to relabel
        num_relabel = self.locals["num_collected_episodes"] + 1

        # for now relabel without caring for skill distribution
        dones = np.where((self.locals["replay_buffer"]).dones == 1)[0]
        print("dones = ", dones)
        for i_episode in range(num_relabel):
            # get start and end idx of episode to relabel
            if dones.shape[0] < (num_relabel + 1) - i_episode:
                start_idx = 0
            else:
                start_idx = dones[-(num_relabel + 1) + i_episode] + 1

            end_idx = dones[-(num_relabel) + i_episode]
            print(f"start_idx = {start_idx}, end_idx = {end_idx}")

            init_empty = (self.locals["replay_buffer"]).next_observations["init_empty"][end_idx]
            out_empty = (self.locals["replay_buffer"]).next_observations["curr_empty"][end_idx]
            old_skill = (self.locals["replay_buffer"]).next_observations["skill"][end_idx]
            print("old skill = ", old_skill)

            # get skill that maximizes reward
            # TODO: relabeling for now assumes that we terminated on change of symbolic state
            if self.relabel:
                new_skill = self.relabel_buffer["max_skill"][i_episode]

                curr_reward = self.locals["replay_buffer"].rewards[end_idx]

                if not (new_skill == old_skill).all():
                    # relabel policy transitions with 50% probability
                    if np.random.normal() > 0.5:
                        # relabel all transitions in episode
                        new_reward = self.relabel_buffer["max_reward"][i_episode]



                        # replace skill and in all transitions and reward in last transition of episode

                        (self.locals["replay_buffer"]).observations["skill"][start_idx: end_idx + 1] = new_skill
                        (self.locals["replay_buffer"]).rewards[end_idx] = new_reward

                # always relabel fm transition
                self.buffer.push(init_empty, new_skill, out_empty)
            else:
                # always relabel fm transition
                self.buffer.push(init_empty, old_skill, out_empty)

        print("-------------------------------------------------------")

        self.relabel_buffer = {"max_reward": [],
                               "max_skill": []}


