import gymnasium as gym
import argparse
import matplotlib.pyplot as plt

import torch
import numpy as np

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../gymframework/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../")
sys.path.append(mod_dir)

mod_dir = os.path.join(dir, "../../pytorch-soft-actor-critic/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../../pytorch-soft-actor-critic/")
sys.path.append(mod_dir)

from sac import SAC
from puzzle_env_small_skill_conditioned import PuzzleEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# args for env
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--skill', default=0, type=int,
                    help='Enumeration of the skill to train')
parser.add_argument('--vel_steps', default=1, type=int,
                    help='Number of times to apply velocity control in one step of the agent')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')
parser.add_argument('--reward_on_change', action='store_true', default=False,
                    help='Whether to give additional reward when box is pushed')
parser.add_argument('--term_on_change', action='store_true', default=False,
                    help='Terminate on change of symbolic state')
parser.add_argument('--random_init_board', action='store_true', default=False,
                    help='If true, it is not ensured that the skill execution is possible in the initial board configuration')
parser.add_argument('--reward_on_end', action='store_true', default=False,
                    help='Always give a reward on the terminating episode')
parser.add_argument('--snap_ratio', default=4., type=int,
                    help='1/Ratio of when symbolic state changes, if box is pushed')

# args for SAC
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='update coefficient for polyak update (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', action="store_true",
                    help='Automaically adjust α (default: False)')
parser.add_argument('--target_entropy', default=None, type=float,
                    help='target entropy when automatic entropy tuning is true')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', # default=256
                    help='batch size (default: 256)')
parser.add_argument('--num_epochs', type=int, default=200000, metavar='N',
                    help='number of training epochs (default: 200000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_episode', type=int, default=50, metavar='N',
                    help='model updates per episode (default: 50)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Environment
env = PuzzleEnv(path='../Puzzles/slidingPuzzle_1x2.g',
                max_steps=100,
                random_init_board=False,
                verbose=1,
                sparse_reward=False,
                reward_on_change=True,
                term_on_change=False,
                reward_on_end=False,
                snapRatio=args.snap_ratio)

env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# init sac agent with same parameters as stable baseline sac agent
print(env.observation_space.shape)
print(env.action_space.shape)
agent = SAC(env.observation_space.shape[0], env.action_space, args)


obs, _ = env.reset()
actions = []
for _ in range(2000):
    action = agent.select_action(obs)
    actions.append(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# plot the actions as points in a 3D graph
actions = np.array(actions)
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection = '3d')
#ax.plot_wireframe(x, y, z, color='black')
ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Sampled Actions of untrained agent (Implementation of Pran et al.)")
plt.show()
plt.savefig("Exploration_pran.png")

# after 140 epis actor learned skill-conditioned with reward shaping when skill execution is always possible