import unittest
import torch
import numpy as np
from forwardmodel_simple_input.forward_model import ForwardModel
from gymframework.puzzle_env_small_skill_conditioned_parallel_training import PuzzleEnv

# Todo: Check output of forward model
# - which skill is most probable for which change in empty field/symbolic state
# - which skill is most probable when no change in empty field happened

# tests for:
# no relabeling if only one skill was applied in all episodes, because number of times skill
# appears should not change for relabeled episodes
# correct relabeling


# initialize forward model
# load fully trained forward model
fm = ForwardModel(num_skills=2, batch_size=10, learning_rate=0.001)
fm.model.load_state_dict(
        torch.load("../SEADS_SlidingPuzzle/forwardmodel_simple_input/models/best_model_change"))

# load gym env
# load rl environment
env = PuzzleEnv(path='Puzzles/slidingPuzzle_1x2.g',
                max_steps=100,
                fm_path="../SEADS_SlidingPuzzle/forwardmodel_simple_input/models/best_model_change",
                random_init_pos=True,
                random_init_config=True,
                random_init_board=True,
                verbose=1,
                give_sym_obs=False,
                sparse_reward=True,
                reward_on_change=True,
                reward_on_end=True,
                term_on_change=True,
                setback=False)


class MyTestCase(unittest.TestCase):
    def test_relabeling(self):
        # input consists of two episodes where both apply different skill that result in a change in symbolic state,
        # but for both the forward model would have predicted that the other skill was applied
        # the other skill
        state1 = np.zeros(8)
        # we applie skill 0
        state1[6] = 1
        state2 = np.zeros(8)
        state2[7] = 1
        action = np.zeros(3)

        epis = [((state1, action, 0, state1, 1),
                 (state1, action, 0, state1, 1),
                 (state1, action, -1, state1, 0),
                 (np.array([0, 1]), np.array([1, 0]), np.array([1, 0]))),
                ((state2, action, 0, state2, 1),
                 (state2, action, 0, state2, 1),
                 (state2, action, -1, state2, 0),
                 (np.array([1, 0]), np.array([0, 1]), np.array([0, 1])))
                ]
        # in the correct result both episodes should be relabeled
        reward1 = 50 * fm.calculate_reward(np.array([0, 1]), np.array([1, 0]), 1)
        reward2 = 50 * fm.calculate_reward(np.array([1, 0]), np.array([0, 1]), 0)

        correct_rl = [((state2, action, 0, state2, 1),
                       (state2, action, 0, state2, 1),
                       (state2, action, reward1, 0)),
                      ((state1, action, 0, state1),
                       (state1, action, 0, state1),
                       (state1, action, reward2, 0))]

        correct_fm = [(np.array([0, 1]), np.array([0, 1]), np.array([1, 0])),
                      (np.array([1, 0]), np.array([1, 0]), np.array([0, 1]))]

        rl_epis, fm_epis = env.relabeling(epis)

        print(env.relabeling(epis))

        self.assertSequenceEqual(env.relabeling(epis), (correct_rl, correct_fm))  # add assertion here


if __name__ == '__main__':
    # Todo: create inputs
    # Todo: create correct outputs
    # Todo: test whether relabeling gets correct outputs

    seed = 12345
    env.seed(seed)
    env.action_space.seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    states = np.array([[0, 1], [1, 0]])
    skills = np.array([[1, 0], [0, 1]])
    for k, skill in enumerate(skills):
        for t, state in enumerate(states):
            print(f"probability over successor state for skill {k} from state {state} = {fm.get_p_matrix(state, skill)}")
            for i, succ_state in enumerate(states):
                print(f"reward for going with skill {k} from state {state} to state {succ_state} ="
                      f" {fm.calculate_reward(state, succ_state, k, normalize=False)}")
            print("===================================================================================================")


    unittest.main()
