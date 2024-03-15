from stable_baselines3 import SAC
import argparse
import numpy as np

from gymframework.puzzle_env_skill_conditioned_parallel_training import PuzzleEnv
from forwardmodel_simple_input.forward_model import ForwardModel
#import os
#import sys
#dir = os.path.dirname(__file__)
#mod_dir = os.path.join(dir, "../gymframework/")
#mod_dir = os.path.join(dir, "../forwardmodel_simple_input/")
#sys.path.append(mod_dir)
#mod_dir = os.path.join(dir, "../")
#sys.path.append(mod_dir)
#
#print(sys_path)
#
#from puzzle_env_skill_conditioned_parallel_training import PuzzleEnv
#from forward_model import ForwardModel

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', type=str, default="skill_conditioned_2x2",
                    help='custom gym environment')
parser.add_argument('--num_skills', default=8, type=int,
                    help='Number of skills policy should learn')
args = parser.parse_args()

if th.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# for making a video
# first ensure a folder
import os
os.system('mkdir -p z.vid')

if args.env_name.__contains__("1x2"):
    target_entropy = -2.5
    puzzle_path = '../Puzzles/slidingPuzzle_1x2.g'
    puzzle_size = [1, 2]

elif args.env_name.__contains__("2x2"):
    target_entropy = -3.
    puzzle_path = '../Puzzles/slidingPuzzle_2x2.g'
    puzzle_size = [2, 2]
    fm_path = "/home/rosa/Documents/Uni/Masterarbeit/parallel2x2-new-penalty_num_skills2_sparseTrue_relabelingFalse/fm/fm"
    model_path = "/home/rosa/Documents/Uni/Masterarbeit/parallel2x2-new-penalty_num_skills2_sparseTrue_relabelingFalse/model/model_330000_steps"

    # goal state for 2x2 puzzle cannot be random, because then for most initial configs the goal would not be reachable
    init_state = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
    goal_state = np.array([[0, 0, 0, 1],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0]])

elif args.env_name.__contains__("2x3"):
    target_entropy = -3.5
    puzzle_path = '../Puzzles/slidingPuzzle_2x3.g'
    puzzle_size = [2, 3]
    fm_path = "/home/rosa/Documents/Uni/Masterarbeit/checkpoints_many-skills/SEADS/2x3/10-skills/parallelseads_2x3-10skills_num_skills10_relabelingFalse_noveltyTrue_seconbestTrue_seed105399/fm/fm"
    model_path = "/home/rosa/Documents/Uni/Masterarbeit/checkpoints_many-skills/SEADS/2x3/10-skills/parallelseads_2x3-10skills_num_skills10_relabelingFalse_noveltyTrue_seconbestTrue_seed105399/model/model_700000_steps"

    # goal state for 2x2 puzzle cannot be random, because then for most initial configs the goal would not be reachable
    init_state = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1]])

    goal_state = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])

    #goal_state = np.array([[0, 1, 0, 0, 0, 0],
    #                       [1, 0, 0, 0, 0, 0],
    #                       [0, 0, 0, 1, 0, 0],
    #                       [0, 0, 1, 0, 0, 0],
    #                       [0, 0, 0, 0, 1, 0]])


elif args.env_name.__contains__("3x3"):
    target_entropy = -4.
    puzzle_path = '../Puzzles/slidingPuzzle_3x3.g'
    puzzle_size = [3, 3]
    max_steps = 200

# load forward model and policy
fm = ForwardModel(num_skills=args.num_skills, puzzle_size=puzzle_size, batch_size=10, learning_rate=0.001)

fm.model.load_state_dict(th.load(fm_path, weights_only=True))
fm.model.eval()

print(fm.get_full_pred())

env = PuzzleEnv(path=puzzle_path,
                puzzlesize=puzzle_size,
                max_steps=100,
                num_skills=args.num_skills,
                logging=False,
                verbose=1,
                fm_path=fm_path,
                train_fm=False,
                sparse_reward=True,
                reward_on_change=True,
                term_on_change=True,
                reward_on_end=False)

model = SAC.load(model_path, env=env)



# plan skills to go from init to goal state

# execute skill
# do not reset environment after skill execution, but set actor to init z plane above its current position
_, plan = fm.dijkstra(init_state.flatten(), goal_state.flatten())
print(f"plan = {plan}")
print(f"skill = {plan[0]}")
skill_idx = 0
obs, _ = env.reset(skill=plan.pop(0), actor_pos=np.array([0., 0.]), sym_state_in=init_state)
num_steps = 0
while True:
    #env.scene.C.view_pose(np.array([0., 0., 2., 0., 0., 0., 0]))
    #env.scene.C.view()
    #env.scene.C.view_savePng('z.vid/')
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    num_steps += 1
    if terminated or num_steps > 100:
        if len(plan) == 0:
            break
        # do not reset environment, but only set actor pos to init z-plane
        # do replanning
        #_, plan = fm.dijkstra(env.scene.sym_state.flatten(), goal_state.flatten())
        print(f"plan = {plan}")
        num_steps = 0
        obs = env.execution_reset(skill=plan.pop(0))

time.sleep(10.)

del model
env.close()