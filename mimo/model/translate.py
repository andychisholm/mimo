import torch
from mimo.model.model import MimoModel
from mimo.model.loader import MimoDataLoader
from mimo.model.preprocess import convert_instance_to_idx_seq, convert_mimo_instances_to_idx_seq
from mimo.model.preprocess import read_instances
from types import SimpleNamespace

params = {
    'model': 'model.chkpt',
    'vocab': 'dataset.pt',
    'beam_size': 5,
    'batch_size': 16,
    'n_best': 1,
    'cuda': True
}


def iter_decode(instances_path, limit):
    opt = SimpleNamespace(**params)

    data = torch.load(opt.vocab)
    settings = data['settings']

    sequences = []
    for iid, source, targets in zip(*read_instances(instances_path, settings.max_src_seq_len, 10, limit)):
        sequences.append({
            'instance_id': iid,
            'source': source,
            'targets': targets,
            'idx_sequence': convert_instance_to_idx_seq([source], data['dict']['src'])[0]
        })
    sequences = sequences[:limit]

    batched_data = MimoDataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=[s['idx_sequence'] for s in sequences],
        cuda=opt.cuda,
        shuffle=False,
        batch_size=opt.batch_size)

    translator = MimoModel(opt)
    translator.model.eval()

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
