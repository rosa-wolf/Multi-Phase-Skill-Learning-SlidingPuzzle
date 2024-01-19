import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from .mlp import MLP
from .nllloss import NLLLoss_customized

from .visualize_transitions import visualize_transition


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
        self.sym_obs_size = self.width * self.height * self.pieces
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_size = self.pieces + 1 + num_skills
        self.output_size = self.pieces + 1

        self.batch_size = batch_size

        # define model loss and optimizer
        self.model = MLP(self.width, self.height, self.num_skills)
        self.model = self.model.to(self.device)
        # loss: negative log-likelihood - log q(z_T | z_0, k)
        self.criterion = NLLLoss_customized()
        self.criterion = self.criterion.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)#, weight_decay=0.00005)

    def train(self, data):
        """
        :param data: data to train on
        """

        state_batch, skill_batch, next_state_batch = data.sample(batch_size=self.batch_size)

        print(state_batch)
        print(skill_batch)

        x = torch.FloatTensor(state_batch).to(self.device)
        k = torch.FloatTensor(skill_batch).to(self.device)
        y = torch.FloatTensor(next_state_batch).to(self.device)

        # input to network is state AND skill
        x = torch.cat((x, k), 1)

        epoch_loss = 0
        epoch_acc = 0

        #num_batches = math.ceil(data.shape[0]/self.batch_size)
        ## go through all batches
        #for i in range(num_batches):
        #    if i == num_batches - 1:
        #        # if we are in last batch take rest of data
        #        x = data[i * self.batch_size: , : self.input_size]
        #        y = data[i * self.batch_size: , self.input_size:]
        #    else:
        #        # take one batch of data
        #        x = data[i * self.batch_size: (i + 1) * self.batch_size, : self.input_size]
        #        y = data[i * self.batch_size: (i + 1) * self.batch_size, self.input_size:]


        self.model.train()

        x = self._process_input(x)
        y = self._process_input(y)

        if (torch.isnan(x)).any():
            print("input contains nan values")

        # set all gradients to zero
        self.optimizer.zero_grad()

        # get y_pred (multiclass classification)
        y_pred = self.model(x)

        # get alpha (probability of state being 1) from y_pred
        #alpha = self.calculate_alpha(x, y_pred)
        #loss, max_loss, max_ep = self.criterion(y_pred, y)
        loss = self.criterion(y_pred, y)

        #print("loss = ", loss)
        #print("=========================================")
        #print("transition with max loss : ", max_loss)
        #visualize_transition(x[max_ep, :30], x[max_ep, 30:], y[max_ep])
        #print("prediction ", y_pred[max_ep].reshape((5, 6)))
        #print("=========================================")

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

        #return epoch_loss / num_batches, epoch_acc / num_batches
        return epoch_loss, epoch_acc

    def evaluate(self, data, num_epis=5):
        """
        :param data: data to test on
        :param num_epis: number of episodes to evaluate for

        :return: mean of loss and accurracy over all episodes
        """
        epoch_loss = 0
        epoch_acc = 0

        for i in range(num_epis):
            state_batch, skill_batch, next_state_batch = data.sample(batch_size=self.batch_size)

            print(state_batch)
            print(skill_batch)

            x = self._process_input(torch.FloatTensor(state_batch))
            k = self._process_input(torch.FloatTensor(skill_batch))
            y = self._process_input(torch.FloatTensor(next_state_batch))

            # input to network is state AND skill
            x = torch.cat((x, k), 1)

            self.model.eval()

            with torch.no_grad():
                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)
                acc = self.calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / float(num_epis), epoch_acc / float(num_epis)



    def calculate_accuracy(self, y_hat, y):
        """
        Calculates how similar the predicted successor state is to the true one
        Calculates how many predictions are completely correct
        Args:
            :param y: real successor state
            :param alpha: porbability of value being one for each entry of symbolic observation
        """
        # calucalate most likely successor state
        y_pred = y_hat.clone()

        pred = torch.zeros(y_pred.shape)

        pred[torch.arange(pred.shape[0]), torch.argmax(y_pred, axis=1)] = 1

        pred = pred.to(self.device)
        # look how many entries are same as in true successor
        # only say its correct if complete symbolic observations are equal
        true = torch.sum((pred == y).all(axis=1))

        # compute percentage of correctly predicted entries
        return true / y.shape[0] # (y.shape[0] * y.shape[1])


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    #def calculate_alpha(self, x, y_pred):
    #    """
    #    Calculates the probabiliy of a value in the symbolic observation to be one given the input and output of the
    #    mlp
    #    Args:
    #        :param x: input to the mlp (symbolic_observation (flattened), one_hot encoding of skill)
    #        :param y_pred: output of the mlp (shape of flattened symbolic observation)
    #    Returns:
    #        alpha: probability of value of each symbolic observation to be 1
#
    #    """
    #    # get alpha (probability of state being 1) from y_pred
    #    alpha = (1 - x[:, :self.sym_obs_size]) * y_pred + x[:, :self.sym_obs_size] * (1 - y_pred)
    #    # make this into probabilities using softmax
    #    # (for each box probabilities of being in one field should sum up to 1)
    #    old_shape = alpha.shape
    #    alpha = alpha.reshape((alpha.shape[0], self.width * self.height - 1, self.width * self.height))
    #    alpha = torch.softmax(alpha, dim=2)
    #    alpha = alpha.reshape(old_shape)
#
    #    return alpha


    def sym_state_to_input(self, state, one_hot=True):
        """
        Looks up which puzzle field is empty in given symbolic observation
        and returns it either as number or as one-hot encoding

        @param state: symbolic state we want to translate
        @param one_hot: if true return one-hot encoding of empty field

        return: empty field in given symbolic state
        """
        state = np.reshape(state, (self.pieces, self.pieces + 1))
        empty = np.where(np.sum(state, axis=0) == 0)[0][0]

        if one_hot:
            out = np.zeros((self.pieces + 1,))
            out[empty] = 1
            return out

        print("empty = ", empty)
        return empty


    def pred_to_sym_state(self, state, empty):
        """
        Given a previous symbolic state and the current empty field it gives the new symbolic state
        If given empty field a is not empty in the given state, then the box that is on that field a in the
        state will be moved to the empty field b in the state, while the field a will become empty

        @param state: symbolic state we transition from
        @param empty: field that is empty after transitioning
        """

        # look up which field is empty in state
        orig_empty = self.sym_state_to_input(state, one_hot=False)

        # reshape state
        state = np.reshape(state, (self.pieces, self.pieces + 1))
        new_state = state.copy()

        # if empty field changes, state also changes
        if orig_empty != empty:
            # get index of box that changes its field
            box = np.where(state[:, empty] == 1)[0][0]
            new_state[box, empty] = 0
            new_state[box, orig_empty] = 1

        return new_state.flatten()

    def successor(self, state: np.array, skill: np.array) -> np.array:
        """
        Returns successor nodes of a given node
        Args:
            :param state: node to find successor to (symbolic observation, falttened)
            :param skill: skill to apply
        Returns:
            succ: most likely successor when applying skill in state
        """

        """
        ####################################################################
        ## For testing of BFS: take correct successor without using model ##
        ####################################################################
        state = state.copy().reshape((5, 6))
        
        k = np.where(skill == 1)[0][0]
        print("skill to apply = ", k)
        
        # check if skill has effect
        # get empty filed
        print("state shape = ", state.shape)
        empty = np.where(np.sum(state, axis=0) == 0)[0][0]
        if empty == SKILLS[k, 1]:
            # get box we are pushing
            box = np.where(state[:, SKILLS[k, 0]] == 1)[0][0]
            state[box, SKILLS[k, 0]] = 0
            state[box, SKILLS[k, 1]] = 1

        return state.flatten()
        ######################################################################
        """
        init_shape = state.shape
        # concatenate state and skill to get input to mlp
        one_hot_input = self.sym_state_to_input(state)

        succ = self.get_prediction(one_hot_input, skill)

        empty = np.where(succ == 1)[0][0]

        # formulate succesor symbolic state given initial one and knowledge of empty field
        # if new field is empty, then box that was on it is now on previous empty field
        # even if those fields are not neighbors
        new_state = self.pred_to_sym_state(state, empty)

        return new_state

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
        TESTED BY TAKING CORRECT SUCCESSOR INSTEAD BY MODEL PREDICTION: BFS WORKS!!!
        SOLUTION ALSO COMPARED TO ONLINE SOLVER: IT GIVES THE SAME SOLUTION!!!

        Returns sequence of skills to go from start to goal configuration
        as predicted by current forward_model
        Args:
            :param start: start configuration (symbolic state, fLattened)
            :param goal: goal configuration (symbolic state, flattened)
        Returns:
            state_sequence: sequence of symbolic states the forward model predicts it be visited when executing the predicted skills
            skill_sequence: sequence of skills to be executed to go from start to goal configuration
            depth: solution depth
        """
        # is start == goal, then stop now
        # algorithm will fail to
        if (start == goal).all():
            return [], []

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
                # next_state = next_state.cpu().detach().numpy()

                # for devugging visualize every state transition prediction
                # visualize_transition(state, one_hot, next_state)

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

    def _process_input(self, x):
        x = x.to(self.device)
        if self.precision == 'float64':
            x = x.to(torch.float64)
        else:
            x = x.to(torch.float32)

        return x

    def get_full_pred(self):
        """Give the output-matrix over all possible inputs"""

        init_empty = torch.eye(self.pieces + 1)
        skill = torch.eye(self.num_skills)

        input = torch.zeros((self.num_skills, self.pieces + 1, self.pieces + 1 + self.num_skills))
        input[:, :, self.pieces + 1:] = skill[:, None, :]
        input[:, :, :self.pieces + 1] = init_empty

        input = input.reshape((self.num_skills * (self.pieces + 1), self.num_skills + self.pieces + 1))

        input = self._process_input(input)

        with torch.no_grad():
            y_pred = self.model(input)

        y_pred = y_pred.reshape(self.num_skills, self.pieces + 1, self.pieces + 1)

        return y_pred.cpu().detach().numpy()

    def get_p_matrix(self, state: np.array, skill: np.array) -> np.array:
        """
        Given an input to the forward model, it calculates the probabilities over the successor state

        @param state: One-hot encoding of empty field
        @param skill: One-hot encoding of skill
        returns: Probabiliy over succssor empty field
        """

        input = np.concatenate((state, skill))

        input = torch.from_numpy(input)
        input = self._process_input(input)

        input = input[None, :]
        # get probability over successor state
        with torch.no_grad():
            y_pred = self.model(input)

        y_pred = y_pred.reshape((self.pieces + 1,))

        # softmax is already taken in forward call
        #y_pred = torch.softmax(y_pred, dim=0)

        return y_pred

    def get_prediction(self, state: np.array, skill: np.array) -> np.array:
        """
        Calculates most likely successor state

        @param state: One-hot encoding of empty field
        @param skill: One-hot encoding of skill

        returns:
        One-hot encoding of most likely successor empty field
        """

        input = np.concatenate((state, skill))

        input = torch.from_numpy(input)
        input = self._process_input(input)

        input = input[None, :]
        # get probability over successor state

        with torch.no_grad():
            y_pred = self.model(input)

        y_pred = y_pred.reshape((self.pieces + 1,))

        # softmax is already taken in forward call
        # y_pred = torch.softmax(y_pred, dim=0)
        # calculate successor state
        # block is on that field with the highest probability
        empty = torch.argmax(y_pred)

        succ_state = np.zeros(state.shape)
        succ_state[empty] = 1

        return succ_state



    def calculate_reward(self, start, end, k, normalize=True) -> float:
        """
        Calculate the reward, that the skill-conditioned policy optimization gets when it does a successful transition
        from state start to state end using skill k

        R(k) = log q(z_T | z_0, k) / sum_k' q(z_T | z_0, k') + log K

        K: number of states

        Args:
            :param start (z_0) : one-hot encoding of empty field agent starts in
            :param end (z_T): one-hot encoding of empty field agent should end
            :param k: skill agent executes (not as a one_hot_encoding)
            :param normalize: if true we add log K, else we do not add it
        Returns:
            reward: R(k)
        """
        ####################################################
        # go through all skills and get sum of likelihoods #
        ####################################################
        # get model prediction of transitioning from z_0 with each skill
        start = torch.from_numpy(start)
        end = torch.from_numpy(end)
        end = end.to(self.device)

        # get one_hot encoding for all skills
        one_hot = torch.eye(self.num_skills)
        # concatenate with input state z_0
        input = start.repeat(self.num_skills, 1)
        input = torch.concatenate((input, one_hot), axis=1)

        input = self._process_input(input)

        with torch.no_grad():
            y_pred = self.model(input)

        # calculate probability to transition to state z_T for each skill
        # we are only interested in the probability of the correct field being empty
        y_pred = y_pred[:, torch.where(end == 1)[0][0]]

        sum_of_probs = torch.sum(y_pred)

        if normalize:
            return np.log(y_pred[k].item() / sum_of_probs.item()) + np.log(self.num_skills)

        return np.log(y_pred[k].item() / sum_of_probs.item())

    def novelty_bonus(self, start, end, skill, others_only=True) -> float:
        """
        given the transition from start to end, returns

                max_k' q(z_end | z_start, k')

        Args:
            :param start (z_0) : one-hot encoding of empty field agent starts in
            :param end (z_T): one-hot encoding of empty field agent should end
            :skill: as scalar(NOT as one-hot encoding!!!)
            :others_only: whether to only calculate the bonus over skills different from k
        """

        ####################################################
        # go through all skills and get sum of likelihoods #
        ####################################################
        # get model prediction of transitioning from z_0 with each skill
        start = torch.from_numpy(start)
        end = torch.from_numpy(end)
        end = end.to(self.device)

        # get one_hot encoding for all skills
        one_hot = torch.eye(self.num_skills)
        # concatenate with input state z_0
        input = start.repeat(self.num_skills, 1)
        input = torch.concatenate((input, one_hot), axis=1)

        input = self._process_input(input)

        with torch.no_grad():
            y_pred = self.model(input)

        # calculate probability to transition to state z_T for each skill
        # we are only interested in the probability of the correct field being empty
        y_pred = y_pred[:, torch.where(end == 1)[0][0]]

        idx = torch.arange(y_pred.shape[0])

        if others_only:
            y_pred = y_pred[idx != skill]

        # additional bonus if for all other skills fm believes transition to be unlikely
        bonus = - torch.max(torch.log(y_pred))
        return bonus.cpu().detach().numpy()


