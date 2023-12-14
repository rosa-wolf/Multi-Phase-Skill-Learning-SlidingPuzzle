import numpy as np
import torch
import time

import argparse
import datetime
import gym
from copy import deepcopy
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

import importlib.machinery
import importlib.util

from forwardmodel_simple_input.forward_model import ForwardModel
from gymframework.puzzle_env_small_skill_conditioned import PuzzleEnv
from FmReplayMemory import FmReplayMemory

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../pytorch-soft-actor-critic/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../pytorch-soft-actor-critic/")
sys.path.append(mod_dir)

from sac import SAC

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Skill-Conditioned-Policy",
                    help='Custom Gym Environment for sliding puzzle')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--z_cov', type=float, default=10, metavar='G',
                    help='Parameter for inverse cov of optimal z-position function (default: 10)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', # default=256
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')
parser.add_argument('--num_epochs', type=int, default=200000, metavar='N',
                    help='number of training epochs (default: 200000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

SKILLS = np.array([[1, 0], [0, 1]])


def visualize_result(states, skills):
    states = np.array(states)
    states = states.reshape((states.shape[0], 1, 2))

    for i, state in enumerate(states):
        print("| {} | {} |".format(np.where(state[:, 1] == 1)[0],
                                   np.where(state[:, 0] == 1)[0]))
        if i < len(skills):
            print("------------------------------------------")
            print("----> skill: {}, intended effect: {}".format(skills[i], SKILLS[skills[i]]))
            print("------------------------------------------")


def process_episode_batch(episode_batch, agent):
    """
    # Add reward given by forward model to the episodes
    Author: Jan Achterhold
    :param episode_batch:
    :param agent:
    :return:
    """

    # TODO: split episode_batch into state, next_state, skill, reward

    # TODO: compute the reward using the forward model (with reward improvements from paper)

    # TODO: clip the rewards
    processed_episode_list = []
    for episode_idx, original_episode in enumerate(episode_batch):
        episode = deepcopy(original_episode)
        # TODO: add calculated rewards to episode_batch

        # TODO: add flags
    pass


if __name__ == "__main__":

    # load rl environment
    env = PuzzleEnv(path='Puzzles/slidingPuzzle_1x2.g',
                    max_steps=50,
                    old_order=True,
                    random_init_pos=True,
                    random_init_config=True,
                    random_init_board=True,
                    verbose=1,
                    give_sym_obs=False,
                    sparse_reward=False,
                    reward_on_change=True,
                    term_on_change=True,
                    setback=False)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # agent is skill conditioned and thus also gets one-hot encoding of skill in addition to observation
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # load trained policy
    path = "skills/sac_checkpoint_puzzle_env_skill_conditioned_sparse_small_3853"
    agent.load_checkpoint(path)

    # load forward model (untrained)
    fm = ForwardModel(width=2,
                      height=1,
                      num_skills=2,
                      batch_size=10,
                      learning_rate=0.001,
                      precision='float64')

    # One short-term and one long-term memory, we sample episodes from
    recent_memory = FmReplayMemory(50, args.seed)
    buffer_memory = FmReplayMemory(2048, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    num_epochs = 20
    # number of terminating transitions (episodes) we want to collect per epoch of training fm
    num_episodes = 10
    num_skills = 2

    for epoch in range(num_epochs):
        # collect 32 episodes
        start_time = time.monotonic()
        for i_episode in range(num_episodes):
            print(f"============================================\nepisode {i_episode}")
            # get initial state
            episode_reward = 0
            episode_steps = 0
            done = False

            # reset env
            state = env.reset()

            # read out which skill is applied (as one-hot encoding)
            skill = state[6:]
            # init state for forward model is one-hot encoding of empty field and one-hot encoding of skill
            init_state = fm.sym_state_to_input(env.init_sym_state)

            while not done:
                # apply skill until termination (store only first and last symbolic state)
                # termination on change of symbolic observation, or step limit
                # sample action from skill conditioned policy

                action = agent.select_action(state)

                next_state, reward, done, sym_state = env.step(action)

                episode_steps += 1
                episode_reward += reward

                state = next_state

            # get one-hot encoding of empty field of terminal state
            term_state = fm.sym_state_to_input(sym_state)

            # append to buffers
            recent_memory.push(init_state, skill, term_state)
            buffer_memory.push(init_state, skill, term_state)


        # sample episodes from long-term memory
        n_samples = 50
        if len(buffer_memory) >= n_samples:
            episodes = buffer_memory.sample(n_samples)
        else:
            # if buffer is not yet big enough, take all episodes from buffer
            episodes = buffer_memory.sample(len(buffer_memory))
        # take all episodes from recent memory, and sampled episodes from long-term memory
        episodes = (*zip(*episodes), *zip(*recent_memory.sample(len(recent_memory))))
        # reshape episodes sampled from buffer
        episodes = np.array(episodes)
        episodes = episodes.reshape((episodes.shape[0], 6))

        # train fm with data for 4 steps
        for _ in range(4):

            train_loss, train_acc = fm.train(torch.from_numpy(episodes))

            # save model
            if not os.path.exists('models/'):
                os.makedirs('models/')
            path = "models/fm_trained-with-policy"
            print("saving model now")
            # don't save whole model, but only parameters
            torch.save(fm.model.state_dict(), path)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = fm.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

    env.close()

