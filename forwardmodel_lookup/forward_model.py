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
        return np.where(one_hot == 1)[1]

    def add_transition(self, skill, init_empty, out_empty) -> None:
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

    def _accuracy(self, skills, states, next_states):
        pass

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


    def compute_reward(self, skill, init_empty, out_empty):
        # get probabilities to go from init_empty to out_empty for all skills
        probs = np.array((self.num_skills, ))
        probs = self.table[:, init_empty, out_empty] / np.sum(self.table[:, init_empty], axis=1)

        return probs[skill] / np.sum(probs)

