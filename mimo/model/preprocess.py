import gzip
import ujson as json
import torch
from collections import namedtuple
from tqdm import tqdm
import random

from mimo.model.components import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD
from mimo.model.components import BOS, EOS, PAD, UNK


def normalize_relation_name(r):
    return '<' + r.replace(' ', '_') + '>'

relation_types = [
    #'date of birth',
    #'place of birth',
    'sex or gender',
    'country of citizenship',
    #'sport',
    #'given name',
    #'family name',
    'occupation'
]
norm_relation_types = [normalize_relation_name(r) for r in relation_types]

word2idx = {
    BOS_WORD: BOS,
    EOS_WORD: EOS,
    PAD_WORD: PAD,
    UNK_WORD: UNK,
}
for i, r in enumerate(norm_relation_types):
    word2idx[r] = max(word2idx.values()) + 1


def encode_instance(instance, max_src_len, max_tgt_len):
    if 'summary' not in instance or not instance['summary']:
        return []
    if not instance['mentions']:
        return []

    left, span, right = random.choice(instance['mentions'])
    mention = left + span + right

    sources = [mention]
    relations = [(normalize_relation_name(k), v) for k, v in instance['relations'].items() if k in relation_types and v]

    if not relations:
        return []

    pairs = []
    for source in sources:
        for name, target in relations:
            src = source[:max_src_len]
            tgt = target[:max_tgt_len]
            pairs.append((
                instance['_id'],
                [BOS_WORD] + src + [EOS_WORD],
                [BOS_WORD] + tgt + [EOS_WORD]
            ))
    return pairs


def encode_mimo_instance(instance, max_src_len, max_tgt_len):
    if 'summary' not in instance or not instance['summary']:
        return []
    sources = [instance['summary']]
    relations = [(normalize_relation_name(k), v) for k, v in instance['relations'].items() if k in relation_types and v]

    if not relations:
        return []

    pairs = []
    for source in sources:
        src = source[:max_src_len]
        pairs.append((
            instance['_id'],
            [BOS_WORD] + src + [EOS_WORD],
            {name: [BOS_WORD] + target[:max_tgt_len] + [EOS_WORD] for name, target in relations}
        ))

    return pairs


def read_instances(path, max_src_len, max_tgt_len, limit=None):
    iids = []
    src_inst = []
    tgt_inst = []

    print('Loading instances from:', path)

    num_instances = 0
    with gzip.open(path) as f:
        for line in tqdm(f):
            num_instances += 1
            for iid, src, tgt in encode_mimo_instance(json.loads(line), max_src_len, max_tgt_len):
                iids.append(iid)
                src_inst.append(src)
                tgt_inst.append(tgt)
            if limit is not None and num_instances >= limit:
                break

    assert len(iids) == len(src_inst) and len(src_inst) == len(tgt_inst)
    print('[Info] Got {} pairs over {} instances from {}'.format(len(tgt_inst), num_instances, path))
    return iids, src_inst, tgt_inst


def build_vocab_idx(word_insts, min_word_count):
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in word_insts]


def convert_mimo_instances_to_idx_seq(instances, word2idx):
    return [{k:[word2idx[w] if w in word2idx else UNK for w in s] for k, s in inst.items()} for inst in instances]


params = {
    'train_path': 'train.jsonl.gz',
    'valid_path': 'dev.jsonl.gz',
    'save_data': 'dataset.pt',
    'max_src_seq_len': 40,
    'max_tgt_seq_len': 40,
    'max_token_src_seq_len': 40 + 2,
    'max_token_tgt_seq_len': 10 + 2,
    'min_word_count': 5,
    'keep_case': False,
    'share_vocab': True,
    'vocab': None,
}

PreprocessOptions = namedtuple('PreprocessOptions', list(params.keys()))

def main():
    opt = PreprocessOptions(**params)

    # load training set
    _, train_src_word_insts, train_tgt_insts = read_instances(
        opt.train_path,
        opt.max_src_seq_len,
        opt.max_tgt_seq_len,
        50000)

    # load validation set
    _, valid_src_word_insts, valid_tgt_insts = read_instances(
        opt.valid_path,
        opt.max_src_seq_len,
        opt.max_tgt_seq_len,
        10000)

    train_tgt_word_insts = [tokens for inst in train_tgt_insts for tokens in inst.values()]

    # build vocab
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_mimo_instances_to_idx_seq(train_tgt_insts, tgt_word2idx)
    valid_tgt_insts = convert_mimo_instances_to_idx_seq(valid_tgt_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')
