from gymframework.puzzle_env import PuzzleEnv
import numpy as np
import torch
from itertools import permutations
import argparse
import time
import csv

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../pytorch-soft-actor-critic")
sys.path.append(mod_dir)
from sac import SAC


# enumerate skills
SKILLS = np.array([[0, 1], [0, 3], [2, 1], [2, 5], [3, 0], [3, 4], [5, 2],
                   [5, 4], [1, 0], [1, 2], [1, 4], [4, 1], [4, 3], [4, 5]])


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', # default=256
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
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

if __name__ == '__main__':
    #skill = 1

    # get puzzle env and sac agent
    #env = PuzzleEnv('slidingPuzzle.g', random_init_pos=True)
    #env.seed(args.seed)
    #env.action_space.seed(args.seed)
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    # Agent
    #agent = SAC(env.observation_space.shape[0], env.action_space, args)
    # load skill to execute
    #agent.load_checkpoint("skills/skill3")


    # go through all possible initial states where skill execution is possible and where it isnt possible
    # write to file in former (one-hot encoding, init_state, goal_state)
    with open('transitions/transitions.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',')
        for skill in range(14):
            fields = np.array([0, 1, 2, 3, 4, 5])
            # get one-hot encoding of skill
            one_hot = np.zeros((14,))
            one_hot[skill] = 1
            fields_poss = np.delete(fields, SKILLS[skill, 1])
            perms = permutations(fields_poss)
            for order in perms:
                print("order = ", order)
                # reset robot position
                #env.reset()
                # set board in env to initial state
                init_state = np.zeros((5, 6))
                for i in range(5):
                    init_state[i, order[i]] = 1
                # get goal state
                goal_state = init_state.copy()
                start = SKILLS[skill][0]
                goal = SKILLS[skill][1]
                box = np.where(init_state[:, start] == 1)[0][0]
                goal_state[box, start] = 0
                goal_state[box, goal] = 1

                writer.writerow(np.concatenate([init_state.flatten(), one_hot, goal_state.flatten()]))

            # add some transitions where skill has no effect
            count = 0
            while count < 30:
                count += 1
                # pick number where no box is initially
                pick = np.random.choice(fields_poss)
                fields_imposs = np.delete(fields, pick)
                np.random.shuffle(fields_imposs)

                # set board in env to initial state
                # goal state is same as init_state as skill has no effect
                init_state = np.zeros((5, 6))
                for i in range(5):
                    init_state[i, order[i]] = 1

                # write to file
                writer.writerow(np.concatenate([init_state.flatten(), one_hot, init_state.flatten()]))



                #env.scene.sym_state = sym_obs
                #env.scene.set_to_symbolic_state()
                #state = env._get_observation()
                #env._obs = env._get_observation()
                #env._old_obs = env._get_observation()

                #print("init_state = ", state[5:])

                ## execute skill
                #episode_reward = 0
                #done = False
                #while not done:
                #    action = agent.select_action(state, evaluate=True)
                #    next_state, reward, done, _ = env.step(action)
                #    episode_reward += reward

                #    state = next_state

                ## save resulting state, action, state tuple
                #print("episode_reward = ", episode_reward)
                #print("final_state = ", state[5:])
                #time.sleep(10.)
