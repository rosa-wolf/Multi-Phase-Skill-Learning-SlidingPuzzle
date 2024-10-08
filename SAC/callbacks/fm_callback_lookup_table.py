import os

import gymnasium as gym
import numpy as np
import torch
import logging
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
from forwardmodel_lookup.forward_model import ForwardModel

class FmCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self,
                 save_path: str,
                 env,
                 seed: float, memory_size=500,
                 sample_size=40,
                 size=[1, 2],
                 num_skills=2,
                 relabel=False,
                 logging=True,
                 eval_freq=500,
                 verbose=0):

        super().__init__(verbose)
        self.save_path = save_path
        self.env = env  # this is not a copy of the env, but a reference to it
        self.sample_size = sample_size
        self.relabel = relabel
        self.relabel_buffer = {"max_reward": [],
                                "max_skill": [],
                                "episode_length": [],
                                "total_num_steps": []}

        # initialize empty replay memory for fm
        self.buffer = FmReplayMemory(memory_size, seed)
        self.verbose = verbose
        self.num_skills = num_skills
        # initialize forward model (untrained)
        self.fm = ForwardModel(width=size[1],
                          height=size[0],
                          num_skills=self.num_skills)

        self.logging = logging
        if self.logging:
            self.test_acc = []
            self.test_loss = []
            self.eval_freq = eval_freq

        # to keep track of circular replay buffer idx
        self.last_done = None


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

        # minimal number of adjacent fields any field has
        self.min_neighbors = 2
        if size.__contains__(1):
            self.min_neighbors = 1

        # number of fields
        self.num_cells = size[0] * size[1]

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        # save fm
        if self.verbose > 0:
            print("Saving initial model now")
        # don't save whole model, but only parameters
        self.fm.save(self.save_path + "/fm")

        logging.basicConfig(filename=self.save_path + '/logging.log', level=logging.INFO, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    def _on_step(self) -> bool:

        if self.locals["dones"] != 0:
            self.relabel_buffer["max_reward"].append((self.locals["infos"][0])["max_reward"])
            self.relabel_buffer["max_skill"].append((self.locals["infos"][0])["max_skill"])
            self.relabel_buffer["episode_length"].append(((self.locals["infos"][0])["episode"]["l"]))
            self.relabel_buffer["total_num_steps"].append(self.n_calls)

        if self.logging:
            if self.n_calls % self.eval_freq == 0:
                # TODO: change evaluation to work with lookup table
                if len(self.buffer) >= self.sample_size:
                    if self.verbose > 0:
                        print("Evaluate now")
                    test_loss, test_acc = self.fm.evaluate(self.buffer)
                    self.test_acc.append(test_acc)
                    self.test_loss.append(test_loss)
                    np.savez(self.save_path + "log_data", test_acc=test_acc, train_acc=train_acc)

                    if self.verbose > 0:
                        print(f'\tEval Loss: {test_loss:.3f} | Eval Acc: {test_acc * 100:.2f}%')

    def _on_rollout_end(self) -> bool:
        if self.env.starting_epis:
            # Look if forward model finds change in symbolic state probable for at least max_neighbor skills in at least one state
            out = self.fm.get_full_pred()
            # out is shape empty_fields x num_skill x output array
            change_reward_scheme = True
            for i in range(out.shape[0]):
                # set probabilities to not change empty field to zero, as we are only looking at transitions where change happens
                out[i, :, i] = 0
                #print(f"out[{i}] = {out[i]}")
                # we want for each init empty field at least min_neighbors transitions to other (adjacent) fields being probable
                num_change = np.where(out[i] >= 0.8)[0].shape[0]
                #print(f"neighborlist = {self.env.neighborlist[str(i)]}")
                if num_change < len(self.env.neighborlist[str(i)]): #self.min_neighbors:
                        change_reward_scheme = False
            if change_reward_scheme:
                #print("changing reward scheme")
                self.env.starting_epis = False
                logging.info("Changing reward scheme after {0} steps".format(self.n_calls))


        # number of episodes to relabel
        num_relabel = self.locals["num_collected_episodes"] + 1

        # for now relabel without caring for skill distribution
        #dones = np.where((self.locals["replay_buffer"]).dones == 1)[0]

        for i_episode in range(num_relabel):
            ## get start and end idx of episode to relabel
            #if dones.shape[0] < (num_relabel + 1) - i_episode:
            #    start_idx = 0
            #else:
            #    start_idx = dones[-(num_relabel + 1) + i_episode] + 1

            #end_idx = dones[-(num_relabel) + i_episode]

            #print(f"steps = {self.relabel_buffer['total_num_steps']}, epi length = {self.relabel_buffer['episode_length']}")

            start_idx = self.relabel_buffer["total_num_steps"][i_episode] - self.relabel_buffer["episode_length"][i_episode]
            end_idx = self.relabel_buffer["total_num_steps"][i_episode] - 1

            #print(f"before wrapping: start_idx = {start_idx}, end_idx = {end_idx}")
            start_idx = start_idx % self.locals["replay_buffer"].buffer_size
            end_idx = end_idx % self.locals["replay_buffer"].buffer_size

            # wrap around end of replay buffer
            #print(f"start_idx = {start_idx}, end_idx = {end_idx}")

            init_empty = (self.locals["replay_buffer"]).next_observations["init_empty"][end_idx]
            out_empty = (self.locals["replay_buffer"]).next_observations["curr_empty"][end_idx]
            old_skill = (self.locals["replay_buffer"]).next_observations["skill"][end_idx]
            #print(f"init_empty = {init_empty}, out_empty = {out_empty}")

            """"""""""""""""""""""""""""""""""""""""""
            # Wrap around
            if self.relabel:
                # get skill that maximizes reward
                # TODO: relabeling for now assumes that we terminated on change of symbolic state
                new_skill = self.relabel_buffer["max_skill"][i_episode]

                #print(f"old_skill = {old_skill}, new_skill = {new_skill}")
                # never relabel policy transitions
                if not (new_skill == old_skill).all():
                    # relabel policy transitions with 50% probability
                    if np.random.normal() > 0.8:
                        #print("Relabeling RL transitions")
                        # relabel all transitions in episode
                        new_reward = self.relabel_buffer["max_reward"][i_episode]
                        if start_idx > end_idx:
                            # wrap around
                            # Todo: check if skill really has right shape
                            # replace skill and in all transitions and reward in last transition of episode
                            (self.locals["replay_buffer"]).observations["skill"][start_idx:] = new_skill
                            (self.locals["replay_buffer"]).observations["skill"][: end_idx + 1] = new_skill
                            # change skill in next state
                            self.locals["replay_buffer"].next_observations["skill"][start_idx:] = new_skill
                            self.locals["replay_buffer"].next_observations["skill"][: end_idx + 1] = new_skill
                        else:
                            # Todo: check if skill really has right shape
                            # replace skill and in all transitions and reward in last transition of episode
                            (self.locals["replay_buffer"]).observations["skill"][start_idx: end_idx + 1] = new_skill
                            # change skill in next state
                            self.locals["replay_buffer"].next_observations["skill"][start_idx: end_idx + 1] = new_skill

                        (self.locals["replay_buffer"]).rewards[end_idx] = new_reward

                # always relabel fm transition
                # TODO: for lookup tabel add transitions directly to table instead
                self.fm.add_transition(self.fm.one_hot_to_scalar(init_empty),
                                       self.fm.one_hot_to_scalar(new_skill),
                                       self.fm.one_hot_to_scalar(out_empty))
                if self.logging:
                    # we only need the buffer to sample test episodes from when logging
                    self.buffer.push(init_empty.flatten(), new_skill.flatten(), out_empty.flatten())
            else:
                self.fm.add_transition(self.fm.one_hot_to_scalar(init_empty),
                                       self.fm.one_hot_to_scalar(old_skill),
                                       self.fm.one_hot_to_scalar(out_empty))

                # TODO: for lookup tabel add transitions directly to table instead
                if self.logging:
                    self.buffer.push(init_empty.flatten(), old_skill.flatten(), out_empty.flatten())

            self.fm.save(self.save_path + "/fm")


        #print("-------------------------------------------------------")

        self.relabel_buffer = {"max_reward": [],
                               "max_skill": [],
                               "episode_length": [],
                               "total_num_steps": []}


