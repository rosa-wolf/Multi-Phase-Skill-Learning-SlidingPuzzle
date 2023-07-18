import numpy as np
import torch
from itertools import permutations
import argparse
import time
import csv

from visualize_transitions import visualize_transition


# enumerate skills
#SKILLS = np.array([[0, 1], [0, 3], [2, 1], [2, 5], [3, 0], [3, 4], [5, 2],
#                   [5, 4], [1, 0], [1, 2], [1, 4], [4, 1], [4, 3], [4, 5]])

SKILLS = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                   [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])

# list of neighbors for each field from 0 to 5
neighbors = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]


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
    with open('../transitions/transitions.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',')
        for skill in range(14):
            print("skill = ", skill)
            fields = np.array([0, 1, 2, 3, 4, 5])
            # get one-hot encoding of skill
            one_hot = np.zeros((14,))
            one_hot[skill] = 1

            # go through permutations where skill execution is possible
            fields_poss = np.delete(fields, SKILLS[skill, 1])
            perms = permutations(fields_poss)
            for order in perms:
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
                writer.writerow(np.concatenate([init_state.flatten(), one_hot, goal_state.flatten()]))


            # only add transitions where neighboring field we do not want to push to is empty
            for free in neighbors[SKILLS[skill, 0]]:
                if free != SKILLS[skill, 1]:
                    print("free = ", free)
                    fields_poss = np.delete(fields, free)
                    perms = permutations(fields_poss)
                    for order in perms:
                        # set board in env to initial state
                        init_state = np.zeros((5, 6))
                        for i in range(5):
                            init_state[i, order[i]] = 1
                        # get goal state
                        goal_state = init_state.copy()

                        writer.writerow(np.concatenate([init_state.flatten(), one_hot, goal_state.flatten()]))

            # put all possible transitions into training data where initially empty field is the one we want to push from
            fields_imposs = np.delete(fields, SKILLS[skill, 0])
            perms = permutations(fields_imposs)
            for order in perms:
                # set board in env to initial state
                init_state = np.zeros((5, 6))
                for i in range(5):
                    init_state[i, order[i]] = 1
                # goal state similar to initial state
                # write to file
                writer.writerow(np.concatenate([init_state.flatten(), one_hot, init_state.flatten()]))


            # add some additional transitions where skill has no effect
            count = 0
            # take random initial states where skill execution is not possible
            # empty field cannot be the one we want to push from
            choice = np.delete(fields, SKILLS[skill, 1])
            while count < 200:
                count += 1
                # pick number where no box is initially
                pick = np.random.choice(choice)
                print("pick = ", pick)

                fields_imposs = np.delete(fields, pick)
                np.random.shuffle(fields_imposs)
                # set board in env to initial state
                # goal state is same as init_state as skill has no effect
                init_state = np.zeros((5, 6))
                for i in range(5):
                    init_state[i, order[i]] = 1

                # write to file
                writer.writerow(np.concatenate([init_state.flatten(), one_hot, init_state.flatten()]))
                #visualize_transition(init_state.flatten(), one_hot, init_state.flatten())





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
