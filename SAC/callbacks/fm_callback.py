import os

import gymnasium as gym
import numpy as np
import torch
import time

from gymnasium.wrappers import TransformReward

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

    def __init__(self, update_freq: int,
                 save_path: str,
                 env,
                 seed: float, memory_size=500,
                 sample_size=30,
                 size=[1, 2],
                 num_skills=2,
                 relabel=False,
                 logging=True,
                 eval_freq=500,
                 verbose=1):

        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.env = env  # this is not a copy of the env, but a reference to it
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

        self.logging = logging
        if self.logging:
            self.test_acc = []
            self.test_loss = []
            self.eval_freq = eval_freq


        # get the max number of adjacent fields of any grid cell
        self.max_neighbors = 4
        if size == [1, 2] or size == [2, 1]:
            self.max_neighbors = 1
        elif size.__contains__(1):
            self.max_neighbors = 2
        elif size == [2, 2]:
            self.max_neighbors = 2
        elif size.__contains__(2):
            self.max_neighbors = 3

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
            self.relabel_buffer["max_reward"].append((self.locals["infos"][0])["max_reward"])
            self.relabel_buffer["max_skill"].append((self.locals["infos"][0])["max_skill"])

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

        if self.logging:
            if self.n_calls % self.eval_freq == 0:
                if len(self.buffer) >= self.sample_size:
                    if self.verbose:
                        print("Evaluate now")
                    test_loss, test_acc = self.fm.evaluate(self.buffer)
                    self.test_acc.append(test_acc)
                    self.test_loss.append(test_loss)
                    np.savez(self.save_path + "log_data", test_acc=test_acc, train_acc=train_acc)

                    if self.verbose > 0:
                        print(f'\tEval Loss: {test_loss:.3f} | Eval Acc: {test_acc * 100:.2f}%')

    def _on_rollout_end(self) -> bool:
        """
        Update fm
        """

        # Look if forward model finds change in symbolic state probable for at least max_neighbor skills in at least one state
        out = self.fm.get_full_pred()
        # out is a num_skills x emtpy_fields x output array
        # on the diagonal elements are the prob to stay in the initial field => set these to zero
        num_skills_change = 0
        for i in range(out.shape[0]):
            # set probabilities to not change empty field to zero
            np.fill_diagonal(out[i], 0)

            print(f"out[{i}] = {out[i]}")
            # check if the probability to change to any different field is high
            if np.any(out[i] > 0.8):
                num_skills_change += 1
                if num_skills_change >= self.max_neighbors:
                    self.env.starting_epis = False
                    print("changing reward scheme now")
                    break




        # number of episodes to relabel
        num_relabel = self.locals["num_collected_episodes"] + 1

        # for now relabel without caring for skill distribution
        dones = np.where((self.locals["replay_buffer"]).dones == 1)[0]
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
            print(f"init_empty = {init_empty}, out_empty = {out_empty}")

            # get skill that maximizes reward
            # TODO: relabeling for now assumes that we terminated on change of symbolic state
            if self.relabel:
                new_skill = self.relabel_buffer["max_skill"][i_episode]

                if not (new_skill == old_skill).all():
                    # relabel policy transitions with 50% probability
                    if np.random.normal() > 0.5:
                        print("Relabeling RL transitions")
                        # relabel all transitions in episode
                        new_reward = self.relabel_buffer["max_reward"][i_episode]

                        # Todo: check if skill really has right shape
                        # replace skill and in all transitions and reward in last transition of episode
                        (self.locals["replay_buffer"]).observations["skill"][start_idx: end_idx + 1] = new_skill
                        # change skill in next state
                        self.locals["replay_buffer"].next_observations["skill"][start_idx: end_idx + 1] = new_skill
                        (self.locals["replay_buffer"]).rewards[end_idx] = new_reward

                # always relabel fm transition
                self.buffer.push(init_empty.flatten(), new_skill.flatten(), out_empty.flatten())
            else:
                # always relabel fm transition
                self.buffer.push(init_empty.flatten(), old_skill.flatten(), out_empty.flatten())


        print("-------------------------------------------------------")

        self.relabel_buffer = {"max_reward": [],
                               "max_skill": []}


