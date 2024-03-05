from stable_baselines3 import SAC
import argparse
import torch
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../gymframework/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../")
sys.path.append(mod_dir)

from puzzle_env_all_skill_conditioned import PuzzleEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# args for env
parser.add_argument('--env_name', type=str, default="skill_conditioned_2x2",
                    help='custom gym environment')
parser.add_argument('--include_box_pos', action='store_true', default=False,
                    help='Whether coordinates of boxes should be included in observation')
parser.add_argument('--snap_ratio', default=4., type=float,
                    help='1/Ratio of when symbolic state changes, if boxes is pushed')
parser.add_argument('--num_skills', default=8, type=int,
                    help='Number of skills policy should learn')
parser.add_argument('--num_steps', default=100, type=int,
                    help='Number of max episode steps')
parser.add_argument('--neg_dist_reward', action='store_true', default=False,
                    help='Give negative distance reward')
parser.add_argument('--movement_reward', action='store_true', default=False,
                    help='Give negative distance reward')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--reward_on_change', action='store_true', default=False,
                    help='Whether to give additional reward when boxes is pushed')
parser.add_argument('--term_on_change', action='store_true', default=False,
                    help='Terminate on change of symbolic state')
parser.add_argument('--random_init_board', action='store_true', default=False,
                    help='If true, it is not ensured that the skill execution is possible in the initial board configuration')
parser.add_argument('--reward_on_end', action='store_true', default=False,
                    help='Always give a reward on the terminating episode')
# args for SAC
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='update coefficient for polyak update (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', # default=256
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
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Environment
if args.env_name.__contains__("1x2"):
    target_entropy = -2.5
    puzzle_path = '../Puzzles/slidingPuzzle_1x2.g'
    puzzle_size = [1, 2]
    skills = [np.array([[0, 1], [1, 0]])]

elif args.env_name.__contains__("2x2"):
    target_entropy = -3.
    puzzle_path = '../Puzzles/slidingPuzzle_2x2.g'
    puzzle_size = [2, 2]
    #skills = [np.array([[0, 1], [1, 0], [2, 3], [3, 2]]),
    #          np.array([[0, 2], [2, 0], [1, 3], [3, 1]])]

    skills = [np.array([[1, 0]]),
              np.array([[2, 0]]),
              np.array([[0, 1]]),
              np.array([[3, 1]]),
              np.array([[0, 2]]),
              np.array([[3, 2]]),
              np.array([[1, 3]]),
              np.array([[2, 3]])]


elif args.env_name.__contains__("2x3"):
    target_entropy = -3.5
    puzzle_path = '../Puzzles/slidingPuzzle_2x3.g'
    puzzle_size = [2, 3]
    skills = [np.array([[0, 1], [1, 2], [3, 4], [4, 5]]),
              np.array([[1, 0], [2, 1], [5, 4], [4, 3]]),
              np.array([[0, 3], [3, 0], [1, 4], [4, 1], [2, 5], [5, 2]])]

    skills = [np.array([[1, 0]]),
             np.array([[3, 0]]),
             np.array([[0, 1]]),
             np.array([[2, 1]]),
             np.array([[4, 1]]),
             np.array([[1, 2]]),
             np.array([[5, 2]]),
             np.array([[0, 3]]),
             np.array([[4, 3]]),
             np.array([[1, 4]]),
             np.array([[3, 4]]),
             np.array([[5, 4]]),
             np.array([[2, 5]]),
             np.array([[4, 5]])]

elif args.env_name.__contains__("3x3"):
    from puzzle_env_3x3 import PuzzleEnv
    target_entropy = -4.
    puzzle_path = '../Puzzles/slidingPuzzle_3x3.g'
    puzzle_size = [3, 3]
    skills = [np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8]]),
              np.array([[1, 0], [2, 1], [4, 3], [5, 4], [7, 6], [8, 7]]),
              np.array([[0, 3], [3, 6], [1, 4], [4, 7], [2, 5], [5, 8]]),
              np.array([[3, 0], [6, 3], [4, 1], [7, 4], [5, 2], [8, 5]])]
else:
    raise ValueError("You must specify the environment to use")


# use different seed than in training
seed = 98765
env = PuzzleEnv(path=puzzle_path,
                max_steps=100,
                puzzlesize=puzzle_size,
                num_skills=args.num_skills,
                skills=skills,
                verbose=1,
                sparse_reward=args.sparse,
                reward_on_change=True,
                neg_dist_reward=args.neg_dist_reward,
                movement_reward=args.movement_reward,
                term_on_change=True,
                reward_on_end=False,
                dict_obs=True,
                give_coord=True,
                seed=seed)

env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model = SAC.load("/home/rosa/Documents/Uni/Masterarbeit/checkpoints_predefined_2x2/2x3-1skill-gamma95lr001-smallerbox_neg_distFalse_movementFalse_sparseTrue_seed123456/model/model_300000_steps", env=env)

#print(f"mean_reward = {mean_reward}, std_reward = {std_reward}\n==========================\n=========================")
obs, _ = env.reset()
num_steps = 0
for _ in range(5000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    num_steps += 1
    if terminated or truncated or num_steps > 50:
        obs, _ = env.reset()
        num_steps = 0

del model
env.close()

# after 140 epis actor learned skill-conditioned with reward shaping when skill execution is always possible