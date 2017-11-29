import torch
import numpy as np
from mimo.model.components import PAD, BOS, EOS


class Beam(object):
    """ Beam search decoder """
    def __init__(self, size, cuda=False, init=None):
        self.size = size
        self.done = False
        self.init = BOS if init is None else init

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(PAD)]
        self.next_ys[0][0] = self.init

    def get_current_state(self):
        """ Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """ Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def advance(self, word_lk):
        """ update the status and check for finished or not """
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == EOS:
            self.done = True
            self.all_scores.append(self.scores)

        return self.done

    def sort_scores(self):
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """ get decoded sequence for current timestep """
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.init] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps))

        return dec_seq

    def get_hypothesis(self, k):
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return hyp[::-1]


class Greedy(Beam):
    def __init__(self, cuda=False):
        super().__init__(1, cuda)
