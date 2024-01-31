import numpy as np

class ForwardModel():
    def __init__(self,
                 width=3,
                 height=2,
                 num_skills=14,
                 seed=1024):
        self._seed(seed)

        self.width = width
        self.height = height
        self.num_fields = self.width * self.height
        self.num_skills = num_skills

        # initialize probabilities with ones, s.t., each transitions is believed to be equally likely
        self.table = np.ones((self.num_skills, self.num_fields, self.num_fields))

    def _seed(self, seed):
        np.random.seed(seed)
        return [seed]

    def one_hot_to_scalar(self, one_hot):
        print(one_hot)
        return np.where(one_hot == 1)[1]

    def sym_state_to_input(self, state, one_hot=True):
        """
        Looks up which puzzle field is empty in given symbolic observation
        and returns it either as number or as one-hot encoding

        :param state: symbolic state we want to translate
        :param one_hot: if true return one-hot encoding of empty field

        returns: empty field in given symbolic state
        """
        state = np.reshape(state, (self.pieces, self.pieces + 1))
        empty = np.where(np.sum(state, axis=0) == 0)[0][0]

        if one_hot:
            out = np.zeros((self.pieces + 1,))
            out[empty] = 1
            return out

        return empty

    def add_transition(self, init_empty, skill, out_empty) -> None:
        """
        :param skill: skill in range [0, self.num_skills -1]
        :param init_empty: field (not as one_hot encoding)
        :param out_empty: empty field after transitioning (not as one-hot encoding)
        """
        #print(f"skill = {skill}, init_empty = {init_empty}, outempty = {out_empty}")
        # transitions where change occured should be weighted stronger
        if (init_empty == out_empty).all():
            self.table[skill, init_empty, out_empty] += .5
        else:
            self.table[skill, init_empty, out_empty] += 1

    def get_full_pred(self):
        """
        Gets the probabilities for all possible transitions
        """
        norm = np.sum(self.table, axis=2).reshape((self.table.shape[0] * self.table.shape[1]))
        norm = norm[None, :]
        norm = np.tile(norm, (self.table.shape[2], 1)).T
        norm = norm.reshape(self.table.shape)

        pred = self.table / norm
        pred = np.swapaxes(pred, 0, 1)

        return pred

    def evaluate(self, data, num_trans=10):
        """
        Samples data from given buffer and calculates loss and accurracy
        :param data: data in form of ReplayBuffer

        :returns: loss, accurracy
        """

        onehot_state_batch, onehot_skill_batch, onehot_next_state_batch = data.sample(batch_size=num_trans)

        state_batch = self.one_hot_to_scalar(onehot_state_batch)
        skill_batch = self.one_hot_to_scalar(onehot_skill_batch)
        next_state_batch = self.one_hot_to_scalar(onehot_next_state_batch)
        #print(f"out = {next_state_batch}")
        loss = self._nllloss(skill_batch, state_batch, onehot_next_state_batch)

        succ = self.successor(skill_batch, state_batch)

        return loss, (np.where(succ == next_state_batch)[0]).shape[0] / state_batch.shape[0]


    def successor(self, skill_batch, init_empty_batch):
        """
        get most likely successor empty field, given:
        :param skill: applied skill
        :param init_empty: initial emtpy field
        """
        return np.argmax(self.table[skill_batch, init_empty_batch], axis=1)

    def _nllloss(self, skill_batch, init_empty_batch, out_empty_batch):
        """
        all input as one-hot encoding
        """

        #print(f"skills = {skill_batch}\ninit_empty={init_empty_batch}")
        prob = self.table[skill_batch, init_empty_batch] / np.sum(self.table[skill_batch, init_empty_batch], axis=1)[:, None]
        #print(f"prob = {prob}")

        #print(f"out_empty = {out_empty_batch}")
        l = - np.nansum(out_empty_batch * np.log(prob))
        return l / skill_batch.shape[0]

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
        init_state = np.where(start == 1)[0][0]
        end_state = np.where(end == 1)[0][0]

        # get probability of going from init_state to end_state for all skills
        y_pred = self.table[:, init_state, end_state] / np.sum(self.table[:, init_state], axis=1)

        sum_of_probs = np.sum(y_pred)

        if normalize:
            return np.log(y_pred[k] / sum_of_probs) + np.log(self.num_skills)

        return np.log(y_pred[k] / sum_of_probs)

    def novelty_bonus(self, start, end, skill, others_only=True) -> float:
        """
        given the transition from start to end, returns

                max_k' q(z_end | z_start, k')

        Args:
            :param start (z_0) : one-hot encoding of empty field agent starts in
            :param end (z_T): one-hot encoding of empty field ended in
            :skill: skill as scalar(NOT as one-hot encoding!!!)
            :others_only: whether to only calculate the bonus over skills different from k
        """

        ####################################################
        # go through all skills and get sum of likelihoods #
        ####################################################
        init_state = np.where(start == 1)[0][0]
        end_state = np.where(end == 1)[0][0]

        # get of going from init_state to end_state for all skills
        y_pred = self.table[:, init_state, end_state] / np.sum(self.table[:, init_state], axis=1)


        idx = np.arange(y_pred.shape[0])
        if others_only:
            y_pred = y_pred[idx != skill]

        # additional bonus if for all other skills fm believes transition to be unlikely
        bonus = - np.max(np.log(y_pred))
        return bonus

    def save(self, path):
        # save the table
        print("save path = ", path)
        np.save(path, self.table)
    def load(self, path):
        # load a table from a file
        table = np.load(path)
        if not self.table.shape == table.shape:
            raise ValueError(f"Shape of loaded lookup table does not match shape of this instance,\
             loaded shape: {table.shape}, instance shape: {self.table.shape}")

        self.table = table
