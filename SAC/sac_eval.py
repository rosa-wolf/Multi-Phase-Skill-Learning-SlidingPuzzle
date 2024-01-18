import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

import torch
import numpy as np

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../gymframework/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../")
sys.path.append(mod_dir)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# args for env
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--num_skills', default=2, type=int,
                    help='Number of skills to train')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')
parser.add_argument('--reward_on_change', action='store_true', default=False,
                    help='Whether to give additional reward when box is pushed')
parser.add_argument('--neg_dist_reward', action='store_true', default=False,
                    help='Give negative distance reward')
parser.add_argument('--movement_reward', action='store_true', default=False,
                    help='Give negative distance reward')
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
parser.add_argument('--tau', type=float, default=0.1, metavar='G',
                    help='update coefficient for polyak update (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
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

if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'

# load SAC model
#log_dir = "checkpoints/single-push"
#log_dir = "checkpoints/skill-conditioned-always-possible"
#log_dir = "checkpoints/skill-conditioned-sparse"
#log_dir = "checkpoints/init-epis-parallel-sparse"
log_dir = "checkpoints/test"
fm_dir = log_dir + "/fm"


# Environment
match args.env_name:
    case "skill_conditioned_1x2":
        from puzzle_env_small_skill_conditioned import PuzzleEnv
        env = PuzzleEnv(path='../Puzzles/slidingPuzzle_1x2.g',
                        max_steps=100,
                        verbose=1,
                        sparse_reward=True,
                        reward_on_change=True,
                        term_on_change=False,
                        reward_on_end=False,
                        snapRatio=args.snap_ratio)
    case "skill_conditioned_2x2":
        from puzzle_env_2x2_skill_conditioned import PuzzleEnv
        env = PuzzleEnv(path='../Puzzles/slidingPuzzle_2x2.g',
                        max_steps=100,
                        verbose=1,
                        sparse_reward=True,
                        reward_on_change=True,
                        neg_dist_reward=False,
                        term_on_change=False,
                        reward_on_end=False,
                        snapRatio=args.snap_ratio)
    case "skill_conditioned_3x3":
        from puzzle_env_3x3_skill_conditioned import PuzzleEnv

        env = PuzzleEnv(path='../Puzzles/slidingPuzzle_3x3.g',
                        max_steps=100,
                        verbose=1,
                        num_skills=args.num_skills,
                        sparse_reward=args.sparse,
                        reward_on_change=args.reward_on_change,
                        neg_dist_reward=args.neg_dist_reward,
                        movement_reward=args.movement_reward,
                        term_on_change=False,
                        reward_on_end=False,
                        snapRatio=args.snap_ratio)
    case "parallel_1x2":
        from puzzle_env_small_skill_conditioned_parallel_training import PuzzleEnv
        env = PuzzleEnv(path='../Puzzles/slidingPuzzle_1x2.g',
                        max_steps=100,
                        num_skills=args.num_skills,
                        fm_path="/home/rosa/Documents/Uni/Masterarbeit/checkpoints/parallel_1x2_num_skills2_relabelingTrue/fm/fm",
                        verbose=1,
                        sparse_reward=True,
                        reward_on_change=True,
                        term_on_change=True,
                        reward_on_end=False,
                        snapRatio=args.snap_ratio)
    case "parallel_2x2":
        from puzzle_env_2x2_skill_conditioned_parallel_training import PuzzleEnv
        env = PuzzleEnv(path='../Puzzles/slidingPuzzle_2x2.g',
                        max_steps=100,
                        num_skills=args.num_skills,
                        verbose=1,
                        fm_path="/home/rosa/Documents/Uni/Masterarbeit/checkpoints/parallel_2x2_num_skills2_relabelingFalse/fm/fm",
                        sparse_reward=True,
                        reward_on_change=True,
                        term_on_change=True,
                        reward_on_end=args.reward_on_end,
                        snapRatio=args.snap_ratio)

# use different seed than in training
seed = 398199

env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#model = SAC.load("/home/rosa/Documents/Uni/Masterarbeit/checkpoints/skill_conditioned_3x3_neg_distFalse_movementFalse/model/model_3100000_steps", evn=env)

#model = SAC.load("/home/rosa/Documents/Uni/Masterarbeit/checkpoints/parallel_2x2/model/model_287000_steps", env=env)

#model = SAC.load("/home/rosa/Documents/Uni/Masterarbeit/checkpoints/parallel_1x2_num_skills2_relabelingTrue/model/model_400000_steps", env=env)

model = SAC.load("/home/rosa/Documents/Uni/Masterarbeit/checkpoints/skill_conditioned_3x3_num_skills24_neg_distTrue_movementFalse_reward_on_changeFalse_sparseFalse/model/model_50000_steps", env=env)
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)


#print(f"mean_reward = {mean_reward}, std_reward = {std_reward}\n==========================\n=========================")
obs, _ = env.reset(skill=23)
num_steps = 0
for _ in range(5000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    num_steps += 1
    if terminated or truncated or num_steps > 20:
        obs, _ = env.reset(skill=23)
        num_steps = 0

del model
env.close()

# after 140 epis actor learned skill-conditioned with reward shaping when skill execution is always possible