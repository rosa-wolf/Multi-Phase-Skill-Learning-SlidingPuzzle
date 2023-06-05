import torch
import time
import numpy as np
import math
import os

from forward_model import ForwardModel


if __name__ == "__main__":
    # get forward model
    my_forwardmodel = ForwardModel(batch_size=50, learning_rate=0.0002)

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

    test_set_size = 90
    best_valid_loss = float('inf')

    # split data into train and test sets for all epochs
    EPOCHS = math.floor(data.shape[0] / test_set_size)
    print("Epochs = ", EPOCHS)

    for epoch in range(EPOCHS):
        # split all data between train and test set
        test_set = data[epoch * test_set_size: epoch * test_set_size + test_set_size]
        train_set = torch.concatenate((data[: epoch * test_set_size], data[epoch * (test_set_size + 1):]))
        start_time = time.monotonic()

        train_loss, train_acc = my_forwardmodel.train(train_set)
        valid_loss, valid_acc = my_forwardmodel.evaluate(test_set)

        # save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if not os.path.exists('models/'):
                os.makedirs('models/')
            path = "models/best_model"
            # print("Saving model to ", path)
            torch.save({'forward_model_dict': my_forwardmodel.model.state_dict()}, path)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = my_forwardmodel.epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    # test whether path-planning works for very simple problem
    init_state = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0]])
    goal_state = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0, 0]])
    states, skills = my_forwardmodel.breadth_first_search(init_state.flatten(), goal_state.flatten())
    print("skills = ", skills)
    print("states = ", states)
