import numpy as np
import math
import os

class Simulated_k:
    def __init__(self, c=5, n=6000, q=50, train_valid_test_ratio='1:1:1', seed=1234):
        # config
        self.guess = 0.25
        self.c = c
        self.n = n
        self.q = q
        self.seed = seed

        # seed for reproducibility
        np.random.seed(seed)
        self.student_state = np.random.randn(n,c)
        self.question_concept = np.random.randint(0,c, (q,))
        self.question_difficulty = np.random.randn(q)
        self.question_skill_gain = np.random.randn(q) * 0.05 + 0.45

        # response generation
        for ques in range(self.q):
            prob = self.snapshot_for_q(ques)
            resp = np.random.binomial(1,p=prob)
            # self.snapshot.append(prob)
            # increase student skills according to responses
            self.increase_skill(ques,resp)

        # snapshot at last student state
        self.snapshot = [self.snapshot_for_q(ques).tolist() for ques in range(self.q)]
        self.snapshot = np.array(self.snapshot).T
        
        assert (self.snapshot < 1).all() and (self.snapshot > 0).all()
        assert self.snapshot.shape == (n,q)

        # train/valid/test split
        parsed_ratio = [int(e) for e in train_valid_test_ratio.split(':')]
        train_end_index = int(n * parsed_ratio[0] / sum(parsed_ratio))
        valid_end_index = int(n * sum(parsed_ratio[:2]) / sum(parsed_ratio))
        self.train_data = self.snapshot[:train_end_index]
        self.valid_data = self.snapshot[train_end_index:valid_end_index]
        self.test_data = self.snapshot[valid_end_index:]

    def increase_skill(self, q, resp):
        # resp : binary vector of size student #
        corr_gain = self.question_skill_gain[q]
        # incorr_gain = self.question_skill_gain_incorrect[q]

        gain = [corr_gain if r == 1 else 0 for r in resp]
        gain = np.array(gain)
        curr_c = self.question_concept[q]
        self.student_state[:,curr_c] += gain

    def snapshot_for_q(self,q):
        curr_c = self.question_concept[q]
        student_skill = self.student_state[:,curr_c]
        difficulty = self.question_difficulty[q]
        corr_prob = self.guess + (1-self.guess)/(1+np.exp(difficulty - student_skill))
        return corr_prob

    def get_data(self):
        return self.train_data, self.valid_data, self.test_data

if __name__ == "__main__":
    a = Simulated_k(n=6000)
    print(a.data)
    a.save_formatted()
    
    
