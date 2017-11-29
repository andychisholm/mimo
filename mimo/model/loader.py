import random
import numpy as np
import torch
from torch.autograd import Variable
from mimo.model.components import PAD
from mimo.model.preprocess import norm_relation_types


class DataLoader(object):
    def __init__(self, src_word2idx, tgt_word2idx, src_insts=None, tgt_insts=None,
                 cuda=True, batch_size=64, shuffle=True, test=False):

        assert src_insts
        assert len(src_insts) >= batch_size

        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)

        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))

        self._batch_size = batch_size

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        return self._tgt_idx2word

    def shuffle(self):
        if self._tgt_insts:
            paired_insts = list(zip(self._src_insts, self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        """ Get the next batch """
        def pad_to_longest(insts):
            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

            inst_data_tensor = Variable(torch.LongTensor(inst_data), volatile=self.test)
            inst_position_tensor = Variable(torch.LongTensor(inst_position), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src_insts = self._src_insts[start_idx:end_idx]
            src_data, src_pos = pad_to_longest(src_insts)

            if not self._tgt_insts:
                return src_data, src_pos
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                tgt_data, tgt_pos = pad_to_longest(tgt_insts)
                return (src_data, src_pos), (tgt_data, tgt_pos)

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()


class MimoDataLoader(object):
    def __init__(self, src_word2idx, tgt_word2idx, src_insts=None, tgt_insts=None,
                 cuda=True, batch_size=64, shuffle=True, test=False):

        assert src_insts
        assert len(src_insts) >= batch_size

        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)

        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))

        self._batch_size = batch_size

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        return self._tgt_idx2word

    def shuffle(self):
        if self._tgt_insts:
            paired_insts = list(zip(self._src_insts, self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        """ Get the next batch """
        def pad_to_longest(insts):
            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

            inst_data_tensor = Variable(torch.LongTensor(inst_data), volatile=self.test)
            inst_position_tensor = Variable(torch.LongTensor(inst_position), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size
            src_insts = self._src_insts[start_idx:end_idx]

            if not self._tgt_insts:
                return {k: pad_to_longest(src_insts) for k in norm_relation_types}
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]

                tgt_insts_by_k = {}
                src_insts_by_k = {}
                for src, tgts in zip(src_insts, tgt_insts):
                    for k, tgt in tgts.items():
                        tgt_insts_by_k.setdefault(k, []).append(tgt)
                        src_insts_by_k.setdefault(k, []).append(src)

                return {
                    k: (pad_to_longest(src_insts_by_k[k]), pad_to_longest(tgt_insts_by_k[k]))
                    for k in tgt_insts_by_k.keys()
                }

                #for k in norm_relation_types:
                #    insts = insts_by_k.get(k, [])
                #    if len(insts) < len(src_insts):
                #        insts.extend([[]] * (len(src_insts) - len(insts)))
                #    insts_by_k[k] = insts

                #print('INSTANCES', len(src_insts), [len(v) for v in insts_by_k.values()])
                #print(src_insts)
                #print()
                #print(insts_by_k)
                #print()

                #tgt = {k: pad_to_longest(insts) for k, insts in insts_by_k.items()}
                #return (src_data, src_pos), tgt

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()