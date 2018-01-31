import ujson as json
from mimo.model.translate import iter_decodes_by_instance
from tqdm import tqdm
from itertools import takewhile
from collections import defaultdict
import numpy

baseline = {
    "<sex_or_gender>": "male",
    "<date_of_birth>": "2000 01 01",
    "<occupation>": "politician",
    "<given_name>": "john",
    "<country_of_citizenship>": "united states of america",
    "<place_of_birth>": "new york city",
    "<date_of_death>": "2000 01 01",
    "<place_of_death>": "paris",
    "<educated_at>": "harvard university",
    "<sport>": "association football",
    "<member_of_sports_team>": "st . louis cardinals",
    "<position_held>": "united states representative",
    "<award_received>": "guggenheim fellowship",
    "<family_name>": "smith",
    "<participant_of>": "2008 summer olympics",
    "<member_of_political_party>": "democratic party"
}


def iter_instance_decodes(translator, path, limit):
    # todo: re-add progress bar
    for instance_id, results in iter_decodes_by_instance(translator, path, limit):
        sources = [' '.join(r['source'][1:-1]) for r in results]
        targets = {k: ' '.join(t[1:-1]) for k, t in results[0]['targets'].items()}

        decodes = defaultdict(list)
        for source_id, result in enumerate(results):
            for relation, outputs in result['outputs'].items():
                for o in outputs:
                    o['source_id'] = source_id
                    o['decoded'] = ' '.join(takewhile(lambda t: t != '</s>', o['tokens']))
                    decodes[relation].extend(outputs)

        decodes = {k: sorted(vs, key=lambda o: o['score'], reverse=True) for k, vs in decodes.items()}
        yield instance_id, {
            'sources': sources,
            'targets': targets,
            'decodes': decodes
        }


def evaluate_decodes(results):
    metrics = defaultdict(lambda: defaultdict(list))

    for eid, entity_results in results.items():
        for relation in entity_results['targets'].keys():
            target = entity_results['targets'][relation]
            outputs = entity_results['decodes'][relation]

            if target is not None:
                metrics[relation]['base'].append(1.0 if baseline[relation] == target else 0.0)

                ovs = []
                for o in outputs:
                    if o['decoded'] not in ovs:
                        ovs.append(o['decoded'])
                for k in [1, 2, 5, 10]:
                    metrics[relation]['sys@' + str(k)].append(1.0 if target in ovs[:k] else 0.0)

    return metrics


def get_summary_metrics(metrics, system='sys@1'):
    micro = []
    macro = []
    for r, system_scores in metrics.items():
        micro.extend(system_scores[system])
        macro.append(numpy.mean(system_scores[system]))

    return {
        'micro': numpy.mean(micro),
        'macro': numpy.mean(macro)
    }


def print_evaluation(metrics):
    systems = ['base', 'sys@1', 'sys@2', 'sys@5', 'sys@10']

    print('=' * (30 + 3 + 8 * len(systems)))
    print('Relation'.rjust(30), '|', ''.join(s.ljust(8) for s in systems))
    print('=' * (30 + 3 + 8 * len(systems)))
    micro_scores = defaultdict(list)
    macro_scores = defaultdict(list)

    for r, system_scores in metrics.items():
        report = r.rjust(30) + ' | '
        for s, scores in [(s, system_scores[s]) for s in systems]:
            score = numpy.mean(scores)
            micro_scores[s].extend(scores)
            macro_scores[s].append(score)
            report += ('%.1f' % (100 * score)).ljust(8)
        print(report)

    print('_' * (30 + 3 + 8 * len(systems)))
    print('Micro'.rjust(30) + ' | ' + ''.join(('%.1f' % (100 * numpy.mean(micro_scores[s]))).ljust(8) for s in systems))
    print('Macro'.rjust(30) + ' | ' + ''.join(('%.1f' % (100 * numpy.mean(macro_scores[s]))).ljust(8) for s in systems))
