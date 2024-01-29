import numpy as np
from FmReplayMemory import FmReplayMemory
from forwardmodel_lookup.forward_model import ForwardModel

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
NUM_SKILLS = 24

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
        if poss > 0.5:
            # field we want to push to is empty
            input[SKILLS[skill, effect, 1]] = 1
            output[SKILLS[skill, effect, 0]] = 1

        else:
            # random field is empty
            # but not the field that has to be empty to be able to execute the skill
            fields = np.arange(NUM_FIELDS)
            fields = np.delete(fields, SKILLS[skill, :, 1])
            empty = np.random.choice(fields)
            input[empty] = 1
            output[empty] = 1

        #print(f"skill = {skill}, effect = {effect},\n input = {input}\n output = {output}")

        dataset.push(input, k, output)

def gather_datapoint():
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
    if poss > 0.5:
        # field we want to push to is empty
        input[SKILLS[skill, effect, 1]] = 1
        output[SKILLS[skill, effect, 0]] = 1
    else:
        # random field is empty
        # but not a field in which the skill execution is possible
        fields = np.arange(NUM_FIELDS)
        fields = np.delete(fields, SKILLS[skill, :, 1])
        empty = np.random.choice(fields)
        input[empty] = 1
        output[empty] = 1
    #print(f"skill = {skill}, effect = {effect},\n input = {input}\n output = {output}")

    return input, k, output

if __name__ == "__main__":

    EPOCHS = 100
    print("Epochs = ", EPOCHS)

    num_train_data = 100
    num_test_data = 30

    test_data = FmReplayMemory(num_test_data, 98765)
    gather_data_empty(test_data, 1000)

    np.random.seed(12345)
    fm = ForwardModel(width=3,
                      height=3,
                      num_skills=NUM_SKILLS,
                      seed=12345)

    for i in range(EPOCHS):
        # append new data to buffer,
        # get 10 new data points
        for _ in range(300):
            one_hot_input, one_hot_skill, one_hot_output = gather_datapoint()
            one_hot_input = one_hot_input[None, :]
            one_hot_skill = one_hot_skill[None, :]
            one_hot_output = one_hot_output[None, :]
            fm.add_transition(fm.one_hot_to_scalar(one_hot_skill)[0],
                              fm.one_hot_to_scalar(one_hot_input)[0],
                              fm.one_hot_to_scalar(one_hot_output)[0])
        #print(f"table = \n {fm.table}")

        # get accurracy over n randomly sampled transitions
        train_loss, train_acc = fm.evaluate(test_data, num_trans=num_test_data)
        print(f'\tEpoch: {i}, Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')




