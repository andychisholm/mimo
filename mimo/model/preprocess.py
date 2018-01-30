import gzip
import ujson as json
import torch
from tqdm import tqdm
import random
from types import SimpleNamespace
from itertools import chain
from collections import Counter

random.seed(1447)

from mimo.model.components import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD
from mimo.model.components import BOS, EOS, PAD, UNK


targets = [{
    'name': 'given name',
    'max_len': 3,
}, {
    'name': 'family name',
    'max_len': 3
}, {
    'name': 'sex or gender',
    'max_len': 1
}, {
    'name': 'date of birth',
    'max_len': 4
}, {
    'name': 'occupation',
    'max_len': 3
}, {
    'name': 'country of citizenship',
    'max_len': 4
}, {
    'name': 'sport',
    'max_len': 2
}, {
    'name': 'date of death',
    'max_len': 4
}, {
    'name': 'place of birth',
    'max_len': 5
}, {
    'name': 'educated at',
    'max_len': 7
}, {
    'name': 'member of sports team',
    'max_len': 9
}, {
    'name': 'place of death',
    'max_len': 5
}, {
    'name': 'position held',
    'max_len': 9
}, {
    'name': 'participant of',
    'max_len': 8
}, {
    'name': 'member of political party',
    'max_len': 6
}, {
    'name': 'award received',
    'max_len': 10
}]


def normalize_relation_name(r):
    return '<' + r.replace(' ', '_') + '>'

target_config = {normalize_relation_name(t['name']): t for t in targets}


def encode_mimo_instance(instance, max_src_len, max_inputs):
    if 'summary' not in instance or not instance['summary']:
        return []
    if not instance['mentions']:
        return []

    relations = []
    for k, v in instance['relations'].items():
        k = normalize_relation_name(k)
        if k in target_config and v:
            relations.append((k, v))

    if not relations:
        return []

    if False:
        mentions = [m for m in instance['mentions'] if list(chain(*m)) != instance['summary']]
        if not mentions:
            return []

        mentions = random.sample(mentions, min(max_inputs, len(mentions)))
        sources = []
        for left, span, right in mentions:
            sources.append(left + ['|'] + span + ['|'] + right)
    else:
        sources = [instance['summary']]

    pairs = []
    for source in sources:
        src = source[:max_src_len]
        pairs.append((
            instance['_id'],
            [BOS_WORD] + src + [EOS_WORD],
            {name: [BOS_WORD] + target[:target_config[name]['max_len']] + [EOS_WORD] for name, target in relations}
        ))

    return pairs


def read_instances(path, max_src_len, max_inputs, limit=None):
    iids = []
    src_inst = []
    tgt_inst = []

    print('Loading instances from:', path)

    num_instances = 0
    with gzip.open(path) as f:
        for line in tqdm(f):
            num_instances += 1
            for iid, src, tgt in encode_mimo_instance(json.loads(line), max_src_len, max_inputs=max_inputs):
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

    word2idx = {
        BOS_WORD: BOS,
        EOS_WORD: EOS,
        PAD_WORD: PAD,
        UNK_WORD: UNK,
    }
    #for i, k in enumerate(target_config.keys()):
    #    word2idx[k] = max(word2idx.values()) + 1

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count >= min_word_count:
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
    return [{k: [word2idx[k][w] if w in word2idx[k] else UNK for w in s] for k, s in inst.items()} for inst in instances]


params = {
    'train_path': 'train.jsonl.gz',
    'valid_path': 'dev.jsonl.gz',
    'save_data': 'dataset.pt',
    'max_src_seq_len': 35,
    'max_token_src_seq_len': 35 + 2,
    'min_word_count': 5,
    'min_tgt_word_count': 2,
    'keep_case': False,
    'share_vocab': False,
    'vocab': None,
}


def main():
    opt = SimpleNamespace(**params)

    # load training set
    _, train_src_word_insts, train_tgt_insts = read_instances(opt.train_path, opt.max_src_seq_len, 1, None)

    # load validation set
    _, valid_src_word_insts, valid_tgt_insts = read_instances(opt.valid_path, opt.max_src_seq_len, 1, None)

    # build vocab
    if opt.share_vocab:
        train_tgt_word_insts = [tokens for inst in train_tgt_insts for tokens in inst.values()]

        print('[Info] Build shared vocabulary for source and target.')
        word2idx = build_vocab_idx(train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
        src_word2idx = word2idx
        tgt_word2idx = {k: word2idx for k in target_config.keys()}
    else:
        print('[Info] Build vocabulary for source.')
        src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)

        print('[Info] Build vocabulary for targets.')
        tgt_tokens = {}
        for inst in train_tgt_insts:
            for k, tokens in inst.items():
                tgt_tokens.setdefault(k, []).append(tokens)

        tgt_word2idx = {}
        for k in target_config.keys():
            vocab = build_vocab_idx(tgt_tokens[k], opt.min_tgt_word_count)
            tgt_word2idx[k] = vocab
            print(k.rjust(30), len(vocab), len(tgt_tokens[k]))

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
            'tgt': tgt_word2idx
        },
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts
        },
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts
        }
    }

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')
