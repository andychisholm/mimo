import math
import time
import json
import os
from collections import defaultdict
from types import SimpleNamespace
from shutil import copyfile


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from mimo.model.components import PAD
from mimo.model.model import MimoTransformer
from mimo.model.loader import MimoDataLoader
from mimo.model.components.optim import ScheduledOptim
from mimo.model.preprocess import target_config
from mimo.model.model import GenerationModel
from mimo.evaluate import decode_entity_relations, evaluate_decodes, get_summary_metrics


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(PAD).data).sum()

    return loss, n_correct


def train_mimo_epoch(model, training_data, crit, optimizer):
    model.train()
    training_data.shuffle()

    total_loss = defaultdict(float)
    n_total_words = defaultdict(float)
    n_total_correct = defaultdict(float)
    n_total_inst = defaultdict(float)

    for i, batch in enumerate(tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False)):
        # forward
        optimizer.zero_grad()

        src, tgts = batch
        preds = model(src, tgts)

        agg_loss = None
        agg_n_correct = None
        for k, pred in preds.items():
            gold = tgts[k][1][0][:, 1:]
            loss, n_correct = get_performance(crit[k], pred, gold)

            # note keeping
            n_words = gold.data.ne(PAD).sum()
            n_total_words[k] += n_words
            n_total_correct[k] += n_correct
            n_total_inst[k] += len(tgts[k][0])
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


        if False:
            """
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
            """

    total_loss['ALL'] = sum(total_loss.values())
    n_total_correct['ALL'] = sum(n_total_correct.values())
    n_total_words['ALL'] = sum(n_total_words.values())
    n_total_inst['ALL'] = sum(n_total_inst.values())

    return {k: (
        total_loss[k]/n_total_words[k],
        n_total_correct[k]/n_total_words[k],
        n_total_words[k],
        n_total_inst[k]
    ) for k in total_loss.keys()}


def eval_mimo_epoch(model):
    model.eval()
    start_time = time.time()

    generator = GenerationModel(model, beam_size=5, n_best=10)
    decodes = decode_entity_relations(generator, 'dev.jsonl.gz', 512)
    metrics = get_summary_metrics(evaluate_decodes(decodes))
    return {
        'micro': metrics['micro'],
        'macro': metrics['macro'],
        'timestamp': time.time(),
        'elapsed': time.time() - start_time
    }


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

    valid_history = []
    persisted_models = set()
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_stats = train_mimo_epoch(model, training_data, crit, optimizer)
        print('### Train - elapsed: {elapsed:3.1f} min, lr={lr} '.format(elapsed=((time.time() - start) / 60), lr=('%.1E'%optimizer.get_current_lr())))
        for k, (train_loss, train_accu, num_train_word, num_train_inst) in sorted(train_stats.items(), key=lambda kv: kv[0]):
            print('{relation}: ppl: {ppl: 8.2f}, acc: {accu:3.2f} %, num: {num_inst:.0f}, tks: {num_words:.0f}'.format(
                relation=k.rjust(30),
                ppl=math.exp(min(train_loss, 100)),
                num_words=num_train_word,
                num_inst=num_train_inst,
                accu=100*train_accu))

        valid_metrics = eval_mimo_epoch(model)
        print('### Valid - elapsed: {elapsed:3.1f} min'.format(elapsed=valid_metrics['elapsed']/60))
        print('          - Micro: %.3f' % (valid_metrics['micro']*100))
        print('          - Macro: %.3f' % (valid_metrics['macro']*100))

        model_name = opt.save_model + '_{score}_{epoch}.chkpt'.format(
            epoch=epoch_i, score=int(10000 * valid_metrics['micro']))

        valid_history.append({
            'filename': model_name,
            'tag': opt.save_model,
            'epoch': epoch_i,
            'metrics': valid_metrics,
        })

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i,
            'config': config,
            'metrics': {
                'validation': valid_metrics
            }
        }
        torch.save(checkpoint, model_name)

        # prune saved models
        persisted_models.add(epoch_i)
        if len(persisted_models) > 5:
            max_metric = None
            max_id = None
            min_metric = None
            min_id = None
            for i in sorted(persisted_models):
                metric = valid_history[i]['metrics']['micro']
                if min_metric is None or metric < min_metric:
                    min_metric = metric
                    min_id = i
                if max_metric is None or metric >= max_metric:
                    max_metric = metric
                    max_id = i
            os.remove(valid_history[min_id]['filename'])
            persisted_models.remove(min_id)
            copyfile(valid_history[max_id]['filename'], opt.save_model + '.chkpt')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))

                log_vf.write(json.dumps(valid_history[-1]) + '\n')


params = {
    'data': 'dataset.pt',
    'epoch': 50,
    'batch_size': 128,

    'd_model': 256,
    'd_word_vec': 256,
    'd_inner_hid': 512,

    'd_k': 64,
    'd_v': 64,

    'n_head': 8,
    'n_layers': 4,
    'n_warmup_steps': 16000,

    'dropout': 0.1,
    'embs_share_weight': False,
    'proj_share_weight': True,
    'log': 'model',
    'save_model': 'bio',
    'save_mode': 'all',
    'no_cuda': False,
    'cuda': True,
    'batches_per_epoch': 1000
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
        cuda=opt.cuda,
        max_iters_per_epoch=params['batches_per_epoch'])  # 1024

    validation_data = MimoDataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['valid']['src'],
        tgt_insts=data['valid']['tgt'],
        batch_size=opt.batch_size,
        shuffle=False,
        test=True,
        cuda=opt.cuda,
        max_iters_per_epoch=64)  # 256

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

    adam = optim.Adam(transformer.get_trainable_parameters(), betas=(0.9, 0.999), eps=1e-09)
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
