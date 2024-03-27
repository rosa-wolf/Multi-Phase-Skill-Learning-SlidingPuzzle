from stable_baselines3 import SAC
import argparse
import torch
import numpy as np
import torch as th
import time
import logging as lg
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from callbacks.fm_callback import FmCallback
from callbacks.my_eval_callback import EvalCallback
from gymframework.puzzle_env_skill_conditioned_parallel_training import PuzzleEnv
from forwardmodel_simple_input.forward_model import ForwardModel

from generate_goal import generate_goal

import os

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# args for env
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--num_skills', default=2, type=int,
                    help='Enumeration of the skill to train')
parser.add_argument('--num_episodes', default=1, type=int,
                    help='Number of episode to collect in each rollout')
parser.add_argument('--vel_steps', default=1, type=int,
                    help='Number of times to apply velocity control in one step of the agent')
parser.add_argument('--relabeling', action='store_true', default=False,
                    help='Do HER for higher level skills')
parser.add_argument('--prior_buffer', action='store_true', default=False,
                    help='Whether to use priority buffer')
parser.add_argument('--second_best', action='store_true', default=False,
                    help='Whether to do second best normalization in calculation of reward')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--doinit', action='store_true', default=False,
                    help='Whether to do initial phase')
parser.add_argument('--dorefinement', action='store_true', default=False,
                    help='Whether to do refinement phase')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')
parser.add_argument('--reward_on_change', action='store_true', default=False,
                    help='Whether to give additional reward when boxes is pushed')
parser.add_argument('--term_on_change', action='store_true', default=False,
                    help='Terminate on change of symbolic state')
parser.add_argument('--random_init_board', action='store_true', default=False,
                    help='If true, it is not ensured that the skill execution is possible in the initial board configuration')
parser.add_argument('--reward_on_end', action='store_true', default=False,
                    help='Always give a reward on the terminating episode')
parser.add_argument('--snap_ratio', default=4., type=int,
                    help='1/Ratio of when symbolic state changes, if boxes is pushed')

# args for SAC
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
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

if args.prior_buffer:
    from Buffer import PriorityDictReplayBuffer
    buffer_class = PriorityDictReplayBuffer
else:
    from Buffer import SeadsBuffer
    buffer_class = SeadsBuffer

log_dir = "checkpoints/" + "parallel" + args.env_name + "_num_skills" + str(args.num_skills) + "_sparse" + str(args.sparse) + "_relabeling" + str(args.relabeling) + "_priorbuffer" + str(args.prior_buffer) + "_seed" + str(args.seed)
os.makedirs(log_dir, exist_ok=True)
fm_dir = log_dir + "/fm"
os.makedirs(fm_dir, exist_ok=True)

max_steps = 100
# Environment
if args.env_name.__contains__("1x2"):
    target_entropy = -2.5
    puzzle_path = '../Puzzles/slidingPuzzle_1x2.g'
    puzzle_size = [1, 2]

elif args.env_name.__contains__("2x2"):
    target_entropy = -3.
    puzzle_path = '../Puzzles/slidingPuzzle_2x2.g'
    puzzle_size = [2, 2]

elif args.env_name.__contains__("2x3"):
    target_entropy = -3.5
    puzzle_path = '../Puzzles/slidingPuzzle_2x3.g'
    fm_path = "/home/rosa/Documents/Uni/Masterarbeit/checkpoints_many-skills/mine/2x3/15-skills/new_refinement/max-sampling/change-03-005/parallel2x3-maxsampling-03-005_num_skills15_sparseTrue_relabelingFalse_priorbufferFalse_seed195738/fm/fm"
    puzzle_size = [2, 3]

    # goal state for 2x2 puzzle cannot be random, because then for most initial configs the goal would not be reachable
    init_state = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0]])

    goal_state = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

elif args.env_name.__contains__("3x3"):
    target_entropy = -4.5
    puzzle_path = '../Puzzles/slidingPuzzle_3x3.g'
    puzzle_size = [3, 3]
    max_steps = 200

else:
    raise ValueError("You must specify the environment to use")

env = PuzzleEnv(path=puzzle_path,
                max_steps=max_steps,
                second_best=args.second_best,
                num_skills=args.num_skills,
                init_phase=args.doinit,
                refinement_phase=args.dorefinement,
                verbose=0,
                fm_path=fm_path,
                puzzlesize=puzzle_size,
                sparse_reward=args.sparse,
                reward_on_change=True,
                term_on_change=True,
                reward_on_end=True,
                relabel=args.relabeling,
                seed=args.seed,
                snapRatio=args.snap_ratio)


# load fm
fm = ForwardModel(num_skills=args.num_skills, puzzle_size=puzzle_size)
fm.model.load_state_dict(th.load(fm_path, weights_only=True))
fm.model.eval()

model = SAC.load("/home/rosa/Documents/Uni/Masterarbeit/checkpoints_many-skills/mine/2x3/15-skills/new_refinement/max-sampling/change-03-005/parallel2x3-maxsampling-03-005_num_skills15_sparseTrue_relabelingFalse_priorbufferFalse_seed195738/model/model_1000000_steps", env=env)

env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


print(fm.get_full_pred())

solution_depth = [1, 2, 3, 4, 5]
successes = {'1': [[], 0, []],
             '2': [[], 0, []],
             '3': [[], 0, []],
             '4': [[], 0, []],
             '5': [[], 0, []]}

for depth in solution_depth:
    for _ in range(50):
        # sample initial sym state
        init_sym_state = env.sample_sym_state()
        goal_sym_state = generate_goal(init_sym_state, depth=depth, puzzle_size=env.scene.puzzlesize)

        print(f"init_sym_state =\n {init_sym_state}")
        print(f"goal_sym_state =\n {goal_sym_state}")

        # execute skill
        # do not reset environment after skill execution, but set actor to init z plane above its current position
        _, plan = fm.dijkstra(init_sym_state.flatten(), goal_sym_state.flatten())
        #_, plan = fm.breadth_first_search(init_sym_state.flatten(), goal_sym_state.flatten())
        #_, plan = fm.breadth_first_search_planner(init_sym_state.flatten(), goal_sym_state.flatten())
        two_ago_plan = None
        one_ago_plan = plan.copy()
        repititions = 0

        print(f"plan = {plan}")
        print(f"skill = {plan[0]}")

        # the puzzle is solvable, thus if the fm returns that it is not this is a failure
        if plan is None:
            successes[str(depth)][0].append(0)
            successes[str(depth)][1] = float(np.sum(np.array((successes[str(depth)][0]))))/float(len(successes[str(depth)][0]))
            successes[str(depth)][2].append(None)
        else:
            successes[str(depth)][2].append(len(plan))
            obs, _ = env.reset(skill=plan.pop(0), actor_pos=np.array([0., 0.]), sym_state_in=init_sym_state)
            num_steps = 0
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                num_steps += 1
                if terminated or num_steps > 100:
                    # do replanning
                    #_, plan = fm.breadth_first_search_planner(env.scene.sym_state.flatten(), goal_sym_state.flatten())
                    #_, plan = fm.dijkstra(env.scene.sym_state.flatten(), goal_sym_state.flatten())
                    if len(plan) == 0:
                        break
                    #print(f"plan = {plan}")
                    #if plan == two_ago_plan:
                    #    print("Plan is equal to two previous plans")
                    #    break
                    two_ago_plan = one_ago_plan
                    one_ago_plan = plan.copy()
                    num_steps = 0

                    obs = env.execution_reset(skill=plan.pop(0))

            if np.all(env.scene.sym_state == goal_sym_state):
                successes[str(depth)][0].append(1)
                successes[str(depth)][1] = float(np.sum(np.array((successes[str(depth)][0])))) / float(
                    len(successes[str(depth)][0]))
            else:
                successes[str(depth)][0].append(0)
                successes[str(depth)][1] = float(np.sum(np.array((successes[str(depth)][0])))) / float(
                    len(successes[str(depth)][0]))
        print(f"successes = {successes}")


time.sleep(10.)



del model
env.close()




