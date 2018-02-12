import torch
from mimo.model.model import GenerationModel
from mimo.model.loader import MimoDataLoader
from mimo.model.preprocess import convert_instance_to_idx_seq, convert_mimo_instances_to_idx_seq
from mimo.model.preprocess import iter_read_instances
from types import SimpleNamespace

params = {
    'model': 'model.chkpt',
    'vocab': 'dataset.pt',
    'batch_size': 32,
    'cuda': True
}


def _iter_decode_batches(translator, config, sequences, cuda, batch_size):
    batched_data = MimoDataLoader(
        config['dict']['src'],
        config['dict']['tgt'],
        src_insts=[s['idx_sequence'] for s in sequences],
        cuda=cuda,
        shuffle=False,
        batch_size=batch_size,
        test=True)

    def _flattened_results(batched_data):
        for batch in batched_data:
            results = translator.translate_batch(batch)

            results_by_instance = []
            for k, (all_hyp, all_scores) in results.items():
                for i, (idx_seqs, scores) in enumerate(zip(all_hyp, all_scores)):
                    if len(results_by_instance) <= i:
                        results_by_instance.append({})
                    results_by_instance[i][k] = list(zip(idx_seqs, scores))

            yield from results_by_instance

    for sequence, decodes_by_relation in zip(sequences, _flattened_results(batched_data)):
        sequence['outputs'] = {}
        for relation, decodes in decodes_by_relation.items():
            for idx_seq, score in decodes:
                sequence['outputs'].setdefault(relation, []).append({
                    'score': score,
                    'tokens': [batched_data.tgt_idx2word[relation][idx] for idx in idx_seq]
                })

        yield sequence


def iter_decode(translator, instances_path, limit, mention_per_instance=5):
    opt = SimpleNamespace(**params)

    config = torch.load(opt.vocab)
    settings = config['settings']
    batch_bound = params['batch_size'] * 10

    num_instances = 0
    sequences = []
    last_iid = None
    for iid, source, targets in iter_read_instances(instances_path, settings.max_src_seq_len, mention_per_instance, limit):
        if iid != last_iid:
            num_instances += 1
            last_iid = iid
            if sequences and len(sequences) >= batch_bound:
                #print("Processing macro batch: %d, (%d / %d)" % (len(sequences), num_instances, limit or 0))
                yield from _iter_decode_batches(translator, config, sequences, opt.cuda, min(opt.batch_size, len(sequences)))
                sequences = []
        sequences.append({
            'instance_id': iid,
            'source': source,
            'targets': targets,
            'idx_sequence': convert_instance_to_idx_seq([source], config['dict']['src'])[0]
        })

    if sequences:
        yield from _iter_decode_batches(translator, config, sequences, opt.cuda, min(opt.batch_size, len(sequences)))
    print('Decoded %d instances' % num_instances)


def iter_decodes_by_instance(*args, **kwargs):
    last_iid = None
    decodes = []
    for r in iter_decode(*args, **kwargs):
        if last_iid is None:
            last_iid = r['instance_id']
        if last_iid != r['instance_id']:
            if decodes:
                yield last_iid, decodes
                decodes = []
            last_iid = r['instance_id']
        decodes.append(r)
    if decodes:
        yield last_iid, decodes
