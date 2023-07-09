import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from mlp import MLP
from nllloss import NLLLoss_customized


"""
- The forward model is a tree, connecting all skills
- Probability of symbolic state being reached, when performing a skill in particular symbolic state 
(effect of skill executions)
- skill conditioned
- multilayer perceptron that predicts weather entry in symbolic observation will flip on skill execution
- trained using gradient descent (using ADAM) (given tuples of sym observation, skill and successor observation)
"""

"""
- skills are actions that lead to transition in symbolic observation
- forward model captures effect of skill on symbolic observation
"""


class ForwardModel(nn.Module):

    """
    - plan sequence of skills from start to goal symbolic state using breadth-first search

    - Model of distribution over z' (successor state)
    Probabilities of [(state, skill) -> succ_state]
    -> MLP predicts probabilities of entries in symbolic observation to flip (p_flip = f(z_0, k))
    -> returns most likely successor state given skill in current state

    Function successor:
    - state that is most probable given skill and model
    - node (board configuration) expansion on symbolic forward model (applying feasible actions)

    Breath-first search
    - begin at goal state
    - nodes through node expansion using successor function
    """
    def __init__(self,
                 width=3,
                 height=2,
                 num_skills=14,
                 seed=1024,
                 batch_size=70,
                 learning_rate=0.001,
                 precision='float32'):
        """
        :param width: width of the game board
        :param height: height of the game board
        """

        super().__init__()

        self.precision = precision  # default is float32
        # set to float64
        if self.precision == 'float64':
            torch.set_default_tensor_type(torch.DoubleTensor)

        # get random seeds for python, numpy and pytorch (for reproducible results)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # get one hot encodings of all skills
        self.skills = np.arange(0, num_skills)

        # calculate size of symbolic observation (to get size of input layer)
        self.width = width
        self.height = height
        # number of puzzle pieces
        self.pieces = self.width * self.height - 1
        self.num_skills = num_skills
        self.sym_obs_size = width * height * self.pieces
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        self.input_size = self.sym_obs_size + num_skills
        self.output_size = self.sym_obs_size

        self.batch_size = batch_size

        # define model loss and optimizer
        self.model = MLP(self.input_size, self.output_size)
        self.model.to(self.device)
        # loss: negative log-likelihood - log q(z_T | z_0, k)
        self.criterion = NLLLoss_customized()
        self.criterion.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)#, weight_decay=0.00005)

    def train(self, data):
        """
        :param data: data to train on
        """

        epoch_loss = 0
        epoch_acc = 0

        num_batches = math.ceil(data.shape[0]/self.batch_size)

        self.model.train()

        # go through all batches
        for i in range(num_batches):
            # if we are in last batch take rest of data
            if i == num_batches - 1:
                x = data[i * self.batch_size: , : self.input_size]
                y = data[i * self.batch_size: , self.input_size:]
            else:
                x = data[i * self.batch_size: i * self.batch_size + self.batch_size, : self.input_size]
                y = data[i * self.batch_size: i * self.batch_size + self.batch_size, self.input_size:]

            x = x.to(self.device)
            y = y.to(self.device)

            if self.precision == 'float64':
                x = x.to(torch.float64)
                y = y.to(torch.float64)
            else:
                x = x.to(torch.float32)
                y = y.to(torch.float64)

            if (torch.isnan(x)).any():
                print("input contains nan values")

            self.optimizer.zero_grad()
            # get y_pred (multiclass classification)
            y_pred = self.model(x)

            # interpret y_pred directly as one-hot encoding of block placements for all blocks
            # do multiclass classification
            y_pred = y_pred.reshape((y_pred.shape[0], self.pieces, self.width * self.height))
            y_pred = torch.softmax(y_pred, axis=2)
            y_pred = y_pred.reshape((y_pred.shape[0], self.pieces * self.width * self.height))

            if (torch.isnan(y_pred)).any():
                print("y_pred contains nan values")

            # get alpha (probability of state being 1) from y_pred
            #alpha = self.calculate_alpha(x, y_pred)

            loss = self.criterion(y_pred, y)


            if torch.isnan(loss).any():
                print("loss is nan")

            loss.backward()
            self.optimizer.step()

            acc = self.calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # check if weights contain nan of inf values
            for p in self.model.parameters():
                if torch.isinf(p).any():
                    print('weights contain inf values, ', p)
                if torch.isnan(p).any():
                    print('weights contain nan values, ', p)

        return epoch_loss / num_batches, epoch_acc / num_batches

    def evaluate(self, data):
        """
        :param data: data to test on
        """
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        num_batches = math.floor(data.shape[0] / self.batch_size)

        with torch.no_grad():
            if num_batches > 0:
                for i in range(num_batches):
                    x = data[i * self.batch_size: i * self.batch_size + self.batch_size, : self.input_size]
                    y = data[i * self.batch_size: i * self.batch_size + self.batch_size, self.input_size:]
                    x = x.to(self.device)
                    y = y.to(self.device)

                    if self.precision == 'float64':
                        x = x.to(torch.float64)
                        y = y.to(torch.float64)
                    else:
                        x = x.to(torch.float32)
                        y = y.to(torch.float64)

                    y_pred = self.model(x)

                    # interpret y_pred directly as one-hot encoding of block placements for all blocks
                    # do multiclass classification
                    y_pred = y_pred.reshape((y_pred.shape[0], self.pieces, self.width*self.height))
                    y_pred = torch.softmax(y_pred, axis=2)
                    y_pred = y_pred.reshape((y_pred.shape[0], self.pieces * self.width * self.height))

                    # get alpha (probability of state being 1) from y_pred
                    # alpha = self.calculate_alpha(x, y_pred)

                    #loss = self.criterion(alpha, y)
                    loss = self.criterion(y_pred, y)

                    acc = self.calculate_accuracy(y_pred, y)

                    epoch_loss += loss.item()
                    epoch_acc += acc.item()

                return epoch_loss / num_batches, epoch_acc / num_batches

            x = data[:, : self.input_size]
            y = data[:, self.input_size: ]
            x = x.to(self.device)
            y = y.to(self.device)
            if self.precision == 'float64':
                x = x.to(torch.float64)
                y = y.to(torch.float64)
            else:
                x = x.to(torch.float32)
                y = y.to(torch.float64)
            y_pred = self.model(x)
            # interpret y_pred directly as one-hot encoding of block placements for all blocks
            # do multiclass classification
            y_pred = y_pred.reshape((y_pred.shape[0], self.pieces, self.width * self.height))
            y_pred = torch.softmax(y_pred, axis=2)
            y_pred = y_pred.reshape((y_pred.shape[0], self.pieces * self.width * self.height))
            # get alpha (probability of state being 1) from y_pred
            #alpha = self.calculate_alpha(x, y_pred)
            loss = self.criterion(y_pred, y)
            acc = self.calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            return epoch_loss, epoch_acc


    def calculate_accuracy(self, alpha, y):
        """
        Calculates how similar the predicted successor state is to the true one
        Calculates how many predictions are completely correct
        Args:
            :param y: real successor state
            :param alpha: porbability of value being one for each entry of symbolic observation
        """
        # TODO: get successor state over max
        # calucalate most likely successor state
        y_hat = torch.bernoulli(alpha)
        # look how many entries are same as in true successor
        # only say its correct if complete symbolic observations are equal
        true = torch.sum((y_hat == y).all(axis=1))

        # compute percentage of correctly predicted entries
        return true / y.shape[0] # (y.shape[0] * y.shape[1])


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def calculate_alpha(self, x, y_pred):
        """
        Calculates the probabiliy of a value in the symbolic observation to be one given the input and output of the
        mlp
        Args:
            :param x: input to the mlp (symbolic_observation (flattened), one_hot encoding of skill)
            :param y_pred: output of the mlp (shape of flattened symbolic observation)
        Returns:
            alpha: probability of value of each symbolic observation to be 1

        """
        # get alpha (probability of state being 1) from y_pred
        alpha = (1 - x[:, :self.sym_obs_size]) * y_pred + x[:, :self.sym_obs_size] * (1 - y_pred)
        # make this into probabilities using softmax
        # (for each box probabilities of being in one field should sum up to 1)
        old_shape = alpha.shape
        alpha = alpha.reshape((alpha.shape[0], self.width * self.height - 1, self.width * self.height))
        alpha = torch.softmax(alpha, dim=2)
        alpha = alpha.reshape(old_shape)

        return alpha

    def successor(self, state, skill):
        """
        Returns successor nodes of a given node
        Args:
            :param state: node to find successor to (symbolic observation)
            :param skill: skill to apply
        Returns:
            succ: most likely successor when applying skill in state
        """
        # TODO: change successor calculation according to multimodal classification

        # concatenate state and skill to get input to mlp
        state = torch.from_numpy(state)
        skill = torch.from_numpy(skill)
        x = torch.concatenate((state.flatten(), skill))
        x = x.to(self.device)
        if self.precision == 'float64':
            x = x.to(torch.float64)
        else:
            x = x.to(torch.float32)

        # do forward pass
        y_pred = self.model(x)
        # interpret y_pred directly as one-hot encoding of block placements for all blocks
        # do multiclass classification
        old_shape = y_pred.shape
        y_pred = y_pred.reshape((self.pieces, self.width * self.height))
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred.reshape(old_shape)

        ## get alpha from flip probability and state
        #alpha = (1 - x[: self.sym_obs_size]) * y_pred + x[:self.sym_obs_size] * (1 - y_pred)
        ## make this into probabilities using softmax
        ## (for each box probabilities of being in one field should sum up to 1)
        #old_shape = alpha.shape
        #alpha = alpha.reshape((self.pieces, self.width * self.height))
        #alpha = torch.softmax(alpha, dim=1)
        #alpha = alpha.reshape(old_shape)

        # calculate successor state
        return torch.bernoulli(y_pred)

    def valid_state(self, state) -> bool:
        """
        Checks whether a symbolic state is valid
        :return: True, if state is valid
        """
        # Check that symbolic state is valid
        # i.e. that sum of each row and each column but one is 1 and that all values are 1 or 0
        col_sum = np.sum(state, axis=0)
        row_sum = np.sum(state, axis=1)

        if np.sum(col_sum) != self.pieces or np.sum(row_sum) != self.pieces:
            return False
        if np.any(col_sum > 1) or np.any(col_sum < 0) or np.any(row_sum > 1) or np.any(row_sum < 0):
            return False

        return True

    def _backtrace(self, start, goal, parent, skill):
        path = [goal]
        skills = []
        while not (path[-1] == start).all():
            key = np.array2string(path[-1]).replace('.', '.').replace('\n', '')
            skills.append(skill[key])
            path.append(parent[key])

        return path[::-1], skills[::-1]

    def breadth_first_search(self, start, goal):
        """
        Returns sequence of skills to go from start to goal configuration
        as predicted by current forward_model
        Args:
            :param start: start configuration (symbolic state)
            :param goal: goal configuration (symbolic state)
        Returns:
            state_sequence: sequence of symbolic states the forward model predicts it be visited when executing the predicted skills
            skill_sequence: sequence of skills to be executed to go from start to goal configuration
            depth: solution depth
        """
        visited = []
        queue = []
        parent = {}
        skill = {}

        # start search from start configuration
        visited.append(start)
        queue.append(start)

        # store parent of start as None to not get key-error
        key = np.array2string(start).replace('.', '').replace('\n', '')
        parent[key] = None

        while queue:
            # go through queue
            state = queue.pop(0)
            # node expansion through applying feasible actions
            for k in self.skills:
                # get one-hot encoding of skill
                one_hot = np.zeros((self.num_skills,))
                one_hot[k] = 1

                # find successor state
                next_state = self.successor(state, one_hot)
                next_state = next_state.cpu().detach().numpy()
                # only append next_state if it is not the same as current state
                if not (state == next_state).all():
                    # look wether next state is a valid symbolic observation
                    if self.valid_state(next_state.reshape((self.pieces, self.width*self.height))):
                        # save transition (state, action -> next_state) tuple
                        # make next state to string to get key for dictionaries
                        key = np.array2string(next_state).replace('.', '').replace('\n', '')

                        # look if successor was already visited when transitioning from A DIFFERENT state
                        not_visited = True
                        if np.any(np.all(next_state == visited, axis=1)):
                            if not (parent[key] == state.astype(int)).all():
                                not_visited = False

                        if not_visited:
                            # record skill used to get to next_state
                            # allow for model to be imperfect/ for multiple skills to lead from
                            # same parent state to same successor state
                            if key not in skill.keys():
                                skill[key] = [k]
                            else:
                                skill[key].append(k)
                            # record parent
                            parent[key] = state.astype(int)
                            # if successor state is goal: break
                            if (next_state == goal).all():
                                # get the state transitions and skills that lead to the goal
                                state_sequence, skill_sequence = self._backtrace(start, goal, parent, skill)
                                return state_sequence, skill_sequence
                            # append successor to visited and queue
                            if not np.any(np.all(next_state == visited, axis=1)):
                                visited.append(next_state)
                                queue.append(next_state)

        print("Goal Not reachable from start configuration")
        return None, None