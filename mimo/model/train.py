import math
import time
from collections import defaultdict
from types import SimpleNamespace

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from mimo.model.components import PAD
from mimo.model.model import MimoTransformer
from mimo.model.loader import MimoDataLoader
from mimo.model.components.optim import ScheduledOptim
from mimo.model.preprocess import target_config


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(PAD).data).sum()

    return loss, n_correct


def train_epoch(model, training_data, crit, optimizer):
    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src, tgt)

        # backward
        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_words = gold.data.ne(PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss/n_total_words, n_total_correct/n_total_words


def eval_epoch(model, validation_data, crit):
    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, pred, gold)

        # note keeping
        n_words = gold.data.ne(PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss/n_total_words, n_total_correct/n_total_words


def train_mimo_epoch(model, training_data, crit, optimizer):
    model.train()
    training_data.shuffle()

    total_loss = defaultdict(float)
    n_total_words = defaultdict(float)
    n_total_correct = defaultdict(float)
    n_total_inst = defaultdict(float)

    max_batches_per_epoch = 1024

    for i, batch in enumerate(tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False)):
        if i == max_batches_per_epoch:
            break
        # forward
        optimizer.zero_grad()

        agg_loss = None
        agg_n_correct = None
        for k, (src, tgt) in batch.items():
            pred = model(src, {k:tgt})[k]

            gold = tgt[0][:, 1:]
            loss, n_correct = get_performance(crit[k], pred, gold)

            # note keeping
            n_words = gold.data.ne(PAD).sum()
            n_total_words[k] += n_words
            n_total_correct[k] += n_correct
            n_total_inst[k] += len(src[0])
            total_loss[k] += loss.data[0]

            if agg_loss is None:
                agg_loss = loss
                agg_n_correct = n_correct
            else:
                agg_loss += loss
                agg_n_correct += n_correct

        agg_loss.backward()
        optimizer.step()
        optimizer.update_learning_rate()


    return {k: (
        total_loss[k]/n_total_words[k],
        n_total_correct[k]/n_total_words[k],
        n_total_words[k],
        n_total_inst[k]
    ) for k in total_loss.keys()}


def eval_mimo_epoch(model, validation_data, crit):
    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
        # prepare data
        agg_loss = None
        agg_n_correct = None
        for k, (src, tgt) in batch.items():
            pred = model(src, {k: tgt})[k]

            gold = tgt[0][:, 1:]
            loss, n_correct = get_performance(crit[k], pred, gold)

            # note keeping
            n_words = gold.data.ne(PAD).sum()
            n_total_words += n_words
            n_total_correct += n_correct
            total_loss += loss.data[0]

            if agg_loss is None:
                agg_loss = loss
                agg_n_correct = n_correct
            else:
                agg_loss += loss
                agg_n_correct += n_correct

    return total_loss/n_total_words, n_total_correct/n_total_words


def train(model, training_data, validation_data, crit, optimizer, opt, config):
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_stats = train_mimo_epoch(model, training_data, crit, optimizer)
        print('### Train - elapsed: {elapse:3.3f} min '.format(elapse=(time.time() - start) / 60))
        for k, (train_loss, train_accu, num_train_word, num_train_inst) in sorted(train_stats.items(), key=lambda kv: kv[0]):
            print('{relation}: ppl: {ppl: 8.2f}, acc: {accu:3.2f} %, num: {num_inst:.0f}, tks: {num_words:.0f}'.format(
                relation=k.rjust(30),
                ppl=math.exp(min(train_loss, 100)),
                num_words=num_train_word,
                num_inst=num_train_inst,
                accu=100*train_accu))


        start = time.time()
        valid_loss, valid_accu = eval_mimo_epoch(model, validation_data, crit)
        print('### Valid - elapsed: {elapse:3.3f} min '.format(elapse=(time.time() - start) / 60))

        print('  - Aggregate')
        print('ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i,
            'config': config
        }

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


params = {
    'data': 'dataset.pt',
    'epoch': 20,
    'batch_size': 64,

    'd_model': 512,
    'd_word_vec': 512,
    'd_inner_hid': 512,

    'd_k': 64,
    'd_v': 64,

    'n_head': 8,
    'n_layers': 6,
    'n_warmup_steps': 25000,

    'dropout': 0.1,
    'embs_share_weight': False,
    'proj_share_weight': True,
    'log': None,
    'save_model': 'model',
    'save_mode': 'best',
    'no_cuda': False,
    'cuda': True
}


def main():
    opt = SimpleNamespace(**params)

    # prepare data
    data = torch.load(opt.data)
    opt.max_token_src_seq_len = data['settings'].max_token_src_seq_len

    print('inst', data['train']['tgt'][0])

    training_data = MimoDataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'],
        batch_size=opt.batch_size,
        cuda=opt.cuda)

    validation_data = MimoDataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['valid']['src'],
        tgt_insts=data['valid']['tgt'],
        batch_size=opt.batch_size,
        shuffle=False,
        test=True,
        cuda=opt.cuda)

    opt.src_vocab_size = training_data.src_vocab_size
    opt.tgt_vocab_sizes = training_data.tgt_vocab_sizes
    print('Target vocab sizes:', training_data.tgt_vocab_sizes)

    if opt.embs_share_weight and any(training_data.src_word2idx != tgt_word2idx for tgt_word2idx in training_data.tgt_word2idx.values()):
        print('[Warning] The src/tgt word2idx table are different but asked to share word embedding.')

    default_decoder_params = {
        'd_model': opt.d_model,
        'd_word_vec': opt.d_word_vec,
        'd_inner_hid': opt.d_inner_hid,
        'n_layers': opt.n_layers // 2,
        'n_head': opt.n_head // 2,
        'dropout': opt.dropout
    }
    decoders = {}
    for k, config in target_config.items():
        decoders[k] = {}
        decoders[k].update(default_decoder_params)
        decoders[k]['n_tgt_vocab'] = opt.tgt_vocab_sizes[k]
        decoders[k].update(config)
    config = {
        'decoders': decoders
    }

    transformer = MimoTransformer(
        opt.src_vocab_size,
        opt.max_token_src_seq_len,
        config['decoders'],
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    adam = optim.Adam(transformer.get_trainable_parameters(), betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(adam, opt.d_model, opt.n_warmup_steps)

    def get_criterion(vocab_size):
        # assign zero weight to PAD tokens
        weight = torch.ones(vocab_size)
        weight[PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = {k: get_criterion(size) for k, size in training_data.tgt_vocab_sizes.items()}
    if opt.cuda:
        transformer = transformer.cuda()
        crit = {k: c.cuda() for k, c in crit.items()}

    train(transformer, training_data, validation_data, crit, optimizer, opt, config)