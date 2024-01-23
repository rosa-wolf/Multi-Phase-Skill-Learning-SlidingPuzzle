import torch
import time
import numpy as np
import math
import os
from forwardmodel_simple_input.visualize_transitions import visualize_transition

#from forwardmodel.forward_model import ForwardModel
from forwardmodel_simple_input.forward_model import ForwardModel

from FmReplayMemory import FmReplayMemory

#SKILLS = np.array([[[1, 0]],
#                   [[3, 0]],
#                   [[0, 1]],
#                   [[2, 1]],
#                   [[4, 1]],
#                   [[1, 2]],
#                   [[5, 2]],
#                   [[0, 3]],
#                   [[4, 3]],
#                   [[1, 4]],
#                   [[3, 4]],
#                   [[5, 4]],
#                   [[2, 5]],
#                   [[4, 5]]])

SKILLS = np.array([[[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8]],
                   [[1, 0], [2, 1], [4, 3], [5, 4], [7, 6], [8, 7]],
                   [[0, 3], [3, 6], [1, 4], [4, 7], [2, 5], [5, 8]],
                   [[3, 0], [6, 3], [4, 1], [7, 4], [5, 2], [8, 5]]])


#SKILLS = np.array([[[1, 0]],
#       [[3, 0]],
#       [[0, 1]],
#       [[2, 1]],
#       [[4, 1]],
#       [[1, 2]],
#       [[5, 2]],
#       [[0, 3]],
#       [[4, 3]],
#       [[6, 3]],
#       [[1, 4]],
#       [[3, 4]],
#       [[5, 4]],
#       [[7, 4]],
#       [[2, 5]],
#       [[4, 5]],
#       [[8, 5]],
#       [[3, 6]],
#       [[7, 6]],
#       [[4, 7]],
#       [[6, 7]],
#       [[8, 7]],
#       [[5, 8]],
#       [[7, 8]]])


NUM_FIELDS = 9
NUM_SKILLS = 4

#SKILLS = np.array([[1, 0], [0, 1]])
def visualize_result(states, skills):
     states = np.array(states)
     states = states.reshape((states.shape[0], NUM_FIELDS -1 , NUM_FIELDS))

     for i, state in enumerate(states):
         print("| {} | {} | {} |\n| {} | {} | {} |".format(np.where(state[:, 2] == 1)[0],
                                                           np.where(state[:, 1] == 1)[0],
                                                           np.where(state[:, 0] == 1)[0],
                                                           np.where(state[:, 5] == 1)[0],
                                                           np.where(state[:, 4] == 1)[0],
                                                           np.where(state[:, 3] == 1)[0]))
         if i < len(skills):
            print("------------------------------------------")
            print("----> skill: {}, intended effect: {}".format(skills[i], SKILLS[skills[i]]))
            print("------------------------------------------")

def gather_data_sym(dataset, num_data):
    for i in range(num_data):
        # gather data
        # sample skill
        num_skills = SKILLS.shape[0]
        k = np.zeros((num_skills))
        skill = np.random.choice(np.arange(num_skills))
        k[skill] = 1

        # sample skill effect
        effect = np.random.choice(np.arange(SKILLS[skill].shape[0]))

        # sample board
        # sample empty field
        # sample whether we want to make sure that skill execution is possible
        poss = np.random.uniform()
        if poss > 0.7:
            # field we want to push to is empty
            empty = SKILLS[skill, effect, 1]
        else:
            # random field is empty
            # but not the field we want to push to
            fields = np.arange(NUM_FIELDS)
            fields = np.delete(fields, SKILLS[skill, effect, 1])
            empty = np.random.choice(fields)

        # occupied fields
        non_empty = np.delete(np.arange(NUM_FIELDS), empty)
        # randomly order puzzle pieces onto fields
        order = np.random.permutation(non_empty)

        # get initial symbolic state
        init_state = np.zeros((NUM_FIELDS - 1, NUM_FIELDS))
        for j in range(NUM_FIELDS - 1):
            init_state[j, order[j]] = 1

        goal_state = init_state.copy()
        # if skill has effect, get correct state transition
        if empty == SKILLS[skill, effect,  1]:
            # get goal state
            start = SKILLS[skill, effect, 0]
            goal = SKILLS[skill, effect,  1]
            box = np.where(init_state[:, start] == 1)[0][0]
            goal_state[box, start] = 0
            goal_state[box, goal] = 1


        dataset.push(init_state.flatten(), k, goal_state.flatten())

def gather_data_empty(dataset, num_data):
    for i in range(num_data):
        # gather data
        # sample skill
        num_skills = SKILLS.shape[0]
        k = np.zeros((num_skills))
        skill = np.random.choice(np.arange(num_skills))
        k[skill] = 1

        # sample skill effect
        effect = np.random.choice(np.arange(SKILLS[skill].shape[0]))

        # sample board
        # sample empty field
        # sample whether we want to make sure that skill execution is possible
        poss = np.random.uniform()
        input = np.zeros((NUM_FIELDS,))
        output = np.zeros((NUM_FIELDS,))
        if poss > 0.7:
            # field we want to push to is empty
            input[SKILLS[skill, effect, 1]] = 1
            output[SKILLS[skill, effect, 0]] = 1

        else:
            # random field is empty
            # but not the field that has to be empty to be able to execute the skill
            fields = np.arange(NUM_FIELDS)
            fields = np.delete(fields, SKILLS[skill, effect, 1])
            empty = np.random.choice(fields)
            input[empty] = 1
            output[empty] = 1

        #print(f"skill = {skill}, effect = {effect},\n input = {input}\n output = {output}")

        dataset.push(input, k, output)


if __name__ == "__main__":

    EPOCHS = 100
    print("Epochs = ", EPOCHS)

    num_train_data = 100
    num_test_data = 20

    train_data = FmReplayMemory(10000, 12345)
    test_data = FmReplayMemory(100000, 98765)

    np.random.seed(98765)
    gather_data_empty(test_data, 100000)
    #gather_data_sym(test_data, 100000)
    np.random.seed(12345)
    gather_data_empty(train_data, 1000)
    #gather_data_sym(train_data, 1000)

    # get forward model
    my_forwardmodel = ForwardModel(width=3,
                                   height=3,
                                   num_skills=NUM_SKILLS,
                                   batch_size=45,
                                   learning_rate=0.01)
    print(my_forwardmodel.model)

    test_loss_list = []
    test_acc_list = []
    train_loss_list = []
    train_acc_list = []

    for epoch in range(EPOCHS):
        # train with all the data in each epoch (no testing for now)
        #test_set = data[epoch * test_set_size: epoch * test_set_size + test_set_size]
        #train_set = torch.concatenate((data[: epoch * test_set_size], data[epoch * test_set_size + test_set_size:]))
        start_time = time.monotonic()

        # append new data to buffer
        gather_data_empty(train_data, num_train_data)
        #gather_data_sym(train_data, num_train_data)

        #print("sample = ", train_data.sample(2))

        # train with data for 4 steps
        for _ in range(1):

            train_loss, train_acc = my_forwardmodel.train(train_data)
            valid_loss, valid_acc = my_forwardmodel.evaluate(test_data)

            # save model
            if not os.path.exists('models/'):

                os.makedirs('models/')
            path = "forwardmodel_simple_input/models/best_model_change"
            print("saving model now")
            # dont save whole model, but only parameters
            torch.save(my_forwardmodel.model.state_dict(), path)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = my_forwardmodel.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(valid_loss)
            test_acc_list.append(valid_acc)

            np.savez("fm_eval_empty_input_4skills", train_loss=train_loss_list, train_acc=train_acc_list, test_loss=test_loss_list, test_acc=test_acc_list)

    # load best model
    #my_forwardmodel.model.load_state_dict(torch.load("models/best_model_change"))
    #my_forwardmodel.model.eval()


    ## test whether path-planning works for very simple problem
    #init_state = np.array([[1, 0, 0, 0, 0, 0],
    #                       [0, 1, 0, 0, 0, 0],
    #                       [0, 0, 1, 0, 0, 0],
    #                       [0, 0, 0, 1, 0, 0],
    #                       [0, 0, 0, 0, 1, 0]])
    ##goal_state = np.array([[1, 0, 0, 0, 0, 0],
    ##                       [0, 0, 1, 0, 0, 0],
    ##                       [0, 0, 0, 0, 0, 1],
    ##                       [0, 0, 0, 1, 0, 0],
    ##                       [0, 1, 0, 0, 0, 0]])
    #goal_state = np.array([[0, 1, 0, 0, 0, 0],
    #                       [0, 0, 0, 1, 0, 0],
    #                       [0, 0, 1, 0, 0, 0],
    #                       [1, 0, 0, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 1]])
    #states, skills = my_forwardmodel.breadth_first_search(init_state.flatten(), goal_state.flatten())
    #print("skills = ", skills)
    #print("states = ", states)
#
    ##visualize_result(states, skills)
