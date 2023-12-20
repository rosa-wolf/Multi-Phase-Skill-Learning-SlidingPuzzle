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
from gymframework.puzzle_env_small_skill_conditioned_parallel_training import PuzzleEnv
from FmReplayMemory import FmReplayMemory

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../pytorch-soft-actor-critic/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../pytorch-soft-actor-critic/")
sys.path.append(mod_dir)

from sac import SAC
from replay_memory import ReplayMemory

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
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--critic_lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.3, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', # default=256
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')
parser.add_argument('--num_epochs', type=int, default=200000, metavar='N',
                    help='number of training epochs (default: 200000)')
parser.add_argument('--num_start_epis', type=int, default=1000, metavar='N',
                    help='number of start episodes where fm is not yet included in training (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--updates_per_epoch', type=int, default=1, metavar='N',
                    help='model updates epoch (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=5000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--automatic_entropy_tuning', action='store_true',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--target_entropy', default=None, type=float,
                    help='target entropy when automatic entropy tuning is true')
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


if __name__ == "__main__":

    # load forward model (untrained)
    fm = ForwardModel(width=2,
                      height=1,
                      num_skills=2,
                      batch_size=100,
                      learning_rate=0.001)

    # save model
    if not os.path.exists('models/'):
        os.makedirs('models/')
    fm_path = "models/fm_trained-with-policy_" + str(args.env_name)
    #fm.model.load_state_dict(torch.load(fm_path))
    print("saving initial model now")
    # don't save whole model, but only parameters
    torch.save(fm.model.state_dict(), fm_path)

    # load rl environment
    env = PuzzleEnv(path='Puzzles/slidingPuzzle_1x2.g',
                    max_steps=100,
                    fm_path=fm_path,
                    random_init_pos=True,
                    random_init_config=True,
                    random_init_board=True,
                    verbose=0,
                    give_sym_obs=False,
                    sparse_reward=False,
                    reward_on_change=True,
                    reward_on_end=False,
                    term_on_change=True,
                    setback=False)

    env.starting_epis = True
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # agent is skill conditioned and thus also gets one-hot encoding of skill in addition to observation
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # load pre-trained policy
    checkpoint_name = args.env_name
    #path = "/home/rosa/Documents/Uni/Masterarbeit/SEADS_SlidingPuzzle/checkpoints/sac_checkpoint_puzzle_env_from_zero"
    #agent.load_checkpoint(path)

    # Training Loop
    # number of terminating transitions (episodes) we want to collect
    num_epochs = args.num_epochs
    num_episodes = 10
    total_num_steps = 0
    total_num_episodes = 0

    ####################################
    #    Parameters for fm training    #
    ####################################
    n_samples_fm = 20

    # One short-term and one long-term memory, we sample episodes from
    # for forward model

    #####################################
    # Parameters for rl policy training #
    #####################################
    policy_updates = 0
    # number of samples to take from long-term memory for policy update
    n_samples_policy = 256

    # replay memory for policy
    recent_memory_policy = ReplayMemory(50000, args.seed)
    buffer_memory_policy = ReplayMemory(1000000, args.seed)
    recent_memory_fm = FmReplayMemory(500, args.seed)
    buffer_memory_fm = FmReplayMemory(5000, args.seed)

    load_memories = False
    if load_memories:
        recent_memory_policy.load_buffer(save_path="checkpoints/sac_memory/recent" + str(args.env_name))
        buffer_memory_policy.load_buffer(save_path= "checkpoints/sac_memory/buffer" + str(args.env_name))
        recent_memory_fm.load_buffer(save_path="checkpoints/fm_memory/recent" + str(args.env_name))
        buffer_memory_fm.load_buffer(save_path="checkpoints/fm_memory/buffer" + str(args.env_name))

    # Tensorboard
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                         checkpoint_name, args.policy,
                                                         "autotune" if args.automatic_entropy_tuning else ""))

    ####################
    # Collect episodes #
    ####################
    for epoch in range(num_epochs):
        # collect 32 episodes
        # storage to gather episodes in which still have to be relabeled
        relabel_episodes = []

        start_time = time.monotonic()
        for i_episode in range(num_episodes):
            total_num_episodes += 1

            if total_num_episodes > args.num_start_epis:
                env.starting_epis = False

            episode_reward = 0
            episode_steps = 0
            terminated = False
            truncated = False

            reset = True
            while reset:
                # reset env
                state = env.reset()

                # read out which skill is applied (as one-hot encoding)
                skill = state[6:]
                print("skill = ", skill)

                # one-hot encoding of empty field
                init_state = fm.sym_state_to_input(env.init_sym_state)

                # look whether the forward model predicts a change in empty field
                y_pred = fm.get_prediction(init_state, skill)
                print(f"state =  {init_state}, y_pred = {y_pred}, p_matrix = {fm.get_p_matrix(init_state, skill)}")

                reset = False

                # only do this if fm is already being trained
                if not env.starting_epis:
                    print("testing for reset")
                    # do reset when state change is predicted as less than 50% probable
                    if (init_state == y_pred).all():
                        # fm model predicts skill execution is not possible
                        # reset the env again with some probability
                        if np.random.uniform() > fm.get_p_matrix(init_state, skill)[np.where(init_state == 1)[0][0]]:
                            reset = True

            # temporary storage to gather one full episode, and the transition (z_0, k, z_T) in
            tmp_episodes = []
            while not (truncated or terminated):
                # apply skill until termination (store only first and last symbolic state)
                # termination on change of symbolic observation, or step limit
                # sample action from skill conditioned policy
                if total_num_episodes < 30:
                    action = env.action_space.sample()
                    # add gaussian noise to action
                    action = np.clip(action + 0.1 * np.random.normal(0, 0.1, 3), -1, 1)
                else:
                    action = agent.select_action(state)

                next_state, reward, terminated, truncated, sym_state = env.step(action)

                print("reward = ", reward)
                # append transition more often if it led to success
                if terminated:
                    print("terminated now")
                    for _ in range(49):
                        # if sym state changed append episode several times
                        recent_memory_policy.push(state, action, reward, next_state, float(not terminated))
                        buffer_memory_policy.push(state, action, reward, next_state, float(not terminated))

                recent_memory_policy.push(state, action, reward, next_state, float(not terminated))
                buffer_memory_policy.push(state, action, reward, next_state, float(not terminated))

                # append to memory for policy training:
                # first append them to list for relabeling
                #tmp_episodes.append((state, action, reward, next_state, float(not terminated)))

                episode_steps += 1
                total_num_steps += 1
                episode_reward += reward
                state = next_state

            print("writing episode reward to file:", episode_reward)
            writer.add_scalar('reward/train', episode_reward, total_num_episodes)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(total_num_episodes, total_num_steps,
                                                                                          episode_steps,
                                                                                          round(episode_reward, 2)))

            # get one-hot encoding of empty field of terminal state
            term_state = fm.sym_state_to_input(sym_state)

            # append to buffers
            recent_memory_fm.push(init_state, skill, term_state)
            buffer_memory_fm.push(init_state, skill, term_state)

            if not (init_state == term_state).all():
                for i in range(10):
                    recent_memory_fm.push(init_state, skill, term_state)
                    buffer_memory_fm.push(init_state, skill, term_state)
            #tmp_episodes.append((init_state, skill, term_state))

            #if not (init_state == term_state).all():
            #    # append episode multiple times if  the sym state changed
            #    for i in range(50):
            #        relabel_episodes.append(tuple(tmp_episodes))
            #else:
            #    relabel_episodes.append(tuple(tmp_episodes))
            # append the complete episode to all episodes
            #recent_memory_fm.push(init_state, skill, term_state)
            #buffer_memory_fm.push(init_state, skill, term_state)

        # give episodes to relabeling and append them to buffers
        #print("relabel_episodes = ", relabel_episodes)
        #rl_epis, fm_epis = env.relabeling(relabel_episodes)

        ## append all episodes to buffers
        #for epi in fm_epis:
        #    recent_memory_fm.push(*epi)
        #    buffer_memory_fm.push(*epi)
        #for epi in rl_epis:
        #    for trans in epi:
        #        recent_memory_policy.push(*trans)
        #        buffer_memory_policy.push(*trans)

        # save memories
        recent_memory_fm.save_buffer(args.env_name, save_path="checkpoints/fm_memory/recent" + str(args.env_name))
        buffer_memory_fm.save_buffer(args.env_name, save_path="checkpoints/fm_memory/buffer" + str(args.env_name))

        recent_memory_policy.save_buffer(args.env_name, save_path="checkpoints/sac_memory/recent" + str(args.env_name))
        buffer_memory_policy.save_buffer(args.env_name, save_path="checkpoints/sac_memory/buffer" + str(args.env_name))

        # only start training the forward model after 200 epochs
        if not env.starting_epis:
            #######################
            # Train Forward Model #
            #######################
            # sample episodes from long-term memory
            if len(buffer_memory_fm) >= n_samples_fm:
                episodes = buffer_memory_fm.sample(n_samples_fm)
            else:
                # if buffer is not yet big enough, take all episodes from buffer
                episodes = buffer_memory_fm.sample(len(buffer_memory_fm))
            # take all episodes from recent memory, and sampled episodes from long-term memory
            episodes = (*zip(*episodes), *zip(*recent_memory_fm.sample(len(recent_memory_fm))))

            # reshape episodes sampled from buffer
            episodes = np.array(episodes)
            episodes = episodes.reshape((episodes.shape[0], 6))
            # train fm with data for 4 steps

            for _ in range(10):
                train_loss, train_acc = fm.train(torch.from_numpy(episodes))

                # don't save whole model, but only parameters
                torch.save(fm.model.state_dict(), fm_path)

                end_time = time.monotonic()
                epoch_mins, epoch_secs = fm.epoch_time(start_time, end_time)
                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

            print("updated forward model 10 times")

        #######################
        #   Train RL Policy   #
        #######################
        # update policy for n steps
        for i in range(args.updates_per_epoch):
            # only do update if we have already gathered enough samples to fill at least one batch
            if len(recent_memory_policy) + n_samples_policy >= args.batch_size and len(buffer_memory_policy) >= n_samples_policy:
                # sample episodes from long-term-memory
                episodes = buffer_memory_policy.sample(n_samples_policy)
                ## take all episodes from recent memory, and sampled episodes from long-term memory
                episodes = (*zip(*episodes), *zip(*recent_memory_policy.sample(len(recent_memory_policy))))
                #episodes = tuple(zip(*episodes))
                #episodes = tuple(np.array(x) for x in episodes)
                #print("episodes + recent = ", episodes)
                episodes = ReplayMemory(len(recent_memory_policy) + n_samples_policy, args.seed, episodes)
                # update networks using sac
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, reward_avg, qf1_avg, qf2_avg, qf_target_avg = agent.update_parameters(
                    episodes, args.batch_size, policy_updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, policy_updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, policy_updates)
                writer.add_scalar('loss/policy', policy_loss, policy_updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, policy_updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, policy_updates)
                writer.add_scalar('avg_batch_reward/reward', reward_avg, policy_updates)
                writer.add_scalar('value/critic1', qf1_avg, policy_updates)
                writer.add_scalar('value/critic2', qf2_avg, policy_updates)
                writer.add_scalar('value/critic_target', qf_target_avg, policy_updates)
                policy_updates += 1

        agent.save_checkpoint("puzzle_env", checkpoint_name)
        print(f"updated policy {args.updates_per_epoch} times")
    env.close()

