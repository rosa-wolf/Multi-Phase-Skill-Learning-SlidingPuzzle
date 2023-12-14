import torch
import time
import numpy as np
import math
import os
from forwardmodel_simple_input.visualize_transitions import visualize_transition

from forwardmodel_simple_input.forward_model import ForwardModel


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


def gather_data(num_data):
    dataset = []
    for i in range(num_data):
        # gather data
        # sample skill
        k = np.zeros((2,))
        skill = np.random.choice(np.arange(2))
        k[skill] = 1
        # sample board
        # sample empty field
        # sample whether we want to make sure that skill execution is possible
        poss = np.random.uniform()
        input = np.zeros((2,))
        output = np.zeros((2,))
        if poss > 0.7:
            # skill execution is possible
            # field we want to push to is empty
            input[SKILLS[skill, 1]] = 1
            output[SKILLS[skill, 0]] = 1
        else:
            # random field (but not the one that has to be for the skill),
            # is empty
            empty = SKILLS[skill, 0]
            input[empty] = 1
            output[empty] = 1

        dataset.append(np.concatenate((input, k, output)))

    dataset = np.array(dataset)
    dataset = torch.from_numpy(dataset)

    return dataset


if __name__ == "__main__":

    EPOCHS = 10
    print("Epochs = ", EPOCHS)

    num_train_data = 50
    num_test_data = 50

    # get forward model
    my_forwardmodel = ForwardModel(width=2,
                                   height=1,
                                   num_skills=2,
                                   batch_size=10,
                                   learning_rate=0.001,
                                   precision='float64')
    print(my_forwardmodel.model)

    for epoch in range(EPOCHS):
        # train with all the data in each epoch (no testing for now)
        #test_set = data[epoch * test_set_size: epoch * test_set_size + test_set_size]
        #train_set = torch.concatenate((data[: epoch * test_set_size], data[epoch * test_set_size + test_set_size:]))
        start_time = time.monotonic()

        train_data = gather_data(num_train_data)
        test_data = gather_data(num_test_data)

        # train with data for 4 steps
        for _ in range(4):

            train_loss, train_acc = my_forwardmodel.train(train_data)
            valid_loss, valid_acc = my_forwardmodel.evaluate(test_data)

            # save model
            if not os.path.exists('forwardmodel_simple_input/models/'):

                os.makedirs('forwardmodel_simple_input/models/')
            path = "forwardmodel_simple_input/models/best_model_change"
            print("saving model now")
            # dont save whole model, but only parameters
            torch.save(my_forwardmodel.model.state_dict(), path)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = my_forwardmodel.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # load best model
    my_forwardmodel.model.load_state_dict(torch.load("models/best_model_change"))
    my_forwardmodel.model.eval()


    # test whether path-planning works for very simple problem
    init_state = np.array([[1, 0]])
    goal_state = np.array([[0, 1]])

    states, skills = my_forwardmodel.breadth_first_search(init_state.flatten(), goal_state.flatten())
    print("skills = ", skills)
    print("states = ", states)

    visualize_result(states, skills)
