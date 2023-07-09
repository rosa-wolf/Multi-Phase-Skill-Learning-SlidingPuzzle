import torch
import time
import numpy as np
import math
import os


from forward_model import ForwardModel

#SKILLS = np.array([[0, 1], [0, 3], [2, 1], [2, 5], [3, 0], [3, 4], [5, 2],
#                   [5, 4], [1, 0], [1, 2], [1, 4], [4, 1], [4, 3], [4, 5]])

SKILLS = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                   [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])
def visualize_result(states, skills):
     states = np.array(states)
     states = states.reshape((states.shape[0], 5, 6))

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


if __name__ == "__main__":
    # load data
    # get training data
    # input of size: sym_obs_size + num_skills
    # output of size: sym_obs_size
    # 120 data points for each skill

    # this is data with initial states where skill has effect
    # as well as transitions where initial and successor state are same because skill has no effect
    train_file = "../transitions/transitions.csv"
    data = torch.from_numpy(np.genfromtxt(train_file, delimiter=","))
    # randomly shuffle rows of data
    data = data[torch.randperm(data.size()[0])]
    best_valid_loss = float('inf')

    test_file = "../transitions/test_transitions.csv"
    test_data = torch.from_numpy(np.genfromtxt(test_file, delimiter=","))
    test_data = test_data[torch.randperm(test_data.size()[0])]

    # split data into train and test sets for all epochs
    EPOCHS = 1000
    print("Epochs = ", EPOCHS)


    # get forward model
    my_forwardmodel = ForwardModel(batch_size=200, learning_rate=0.001, precision='float64')
    print(my_forwardmodel.model)

    for epoch in range(EPOCHS):
        # train with all the data in each epoch (no testing for now)
        #test_set = data[epoch * test_set_size: epoch * test_set_size + test_set_size]
        #train_set = torch.concatenate((data[: epoch * test_set_size], data[epoch * test_set_size + test_set_size:]))
        start_time = time.monotonic()

        train_loss, train_acc = my_forwardmodel.train(data)
        valid_loss, valid_acc = my_forwardmodel.evaluate(test_data)

        # save best model
        #if valid_loss < best_valid_loss:
        #    best_valid_loss = valid_loss
        #    if not os.path.exists('models/'):
        #        os.makedirs('models/')
        #    path = "models/best_model"
        #    # print("Saving model to ", path)
        #    torch.save(my_forwardmodel.model, path)

        # save model
        if not os.path.exists('models/'):
            os.makedirs('models/')
            path = "models/best_model"
            torch.save(my_forwardmodel.model, path)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = my_forwardmodel.epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # load best model
    my_forwardmodel.model = torch.load("models/best_model")
    my_forwardmodel.model.eval()

    # test whether path-planning works for very simple problem
    init_state = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0]])
    #goal_state = np.array([[1, 0, 0, 0, 0, 0],
    #                       [0, 0, 1, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 1],
    #                       [0, 0, 0, 1, 0, 0],
    #                       [0, 1, 0, 0, 0, 0]])
    goal_state = np.array([[0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1]])
    states, skills = my_forwardmodel.breadth_first_search(init_state.flatten(), goal_state.flatten())
    print("skills = ", skills)
    print("states = ", states)

    visualize_result(states, skills)
