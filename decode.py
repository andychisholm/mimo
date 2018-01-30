from mimo.evaluate import decode_entity_relations, evaluate_decodes, print_evaluation
from mimo.model.model import GenerationModel
import ujson as json
from itertools import takewhile
from collections import defaultdict

LIMIT = 1024
VERBOSE=LIMIT <= 128
DECODE = True

MODEL_PATH = 'model.chkpt'
INSTANCES_PATH = 'dev.jsonl.gz'

translator = GenerationModel.load(path='model.chkpt', cuda=True, beam_size=5, n_best=10)

if DECODE:
    with open('output.jsonl', 'w') as f:
        json.dump(decode_entity_relations(translator, INSTANCES_PATH, LIMIT), f)

with open('output.jsonl', 'r') as f:
    decodes = json.load(f)

metrics = evaluate_decodes(decodes)
print_evaluation(metrics)

if False:
    for entity_id, results in decodes.items():
        sources = [' '.join(r['source'][1:-1]) for r in results]
        target_relations = {k: ' '.join(t[1:-1]) for k, t in results[0]['targets'].items()}
        system_relations = defaultdict(list)

        for source_id, result in enumerate(results):
            for relation, outputs in result['outputs'].items():
                for o in outputs:
                    o['source_id'] = source_id
                    o['decoded'] = ' '.join(takewhile(lambda t: t != '</s>', o['tokens']))
                system_relations[relation].extend(outputs)
        system_relations = {k: sorted(vs, key=lambda o: o['score'], reverse=True) for k, vs in system_relations.items()}

        if VERBOSE:
            print()
            print('#####', entity_id, '#####')
            for i, s in enumerate(sources):
                print(str(i)+'>', s)
            print()
            for relation, outputs in system_relations.items():
                target = target_relations.get(relation)
                target = '<null>' if target is None else target
                output = outputs[0]
                print(relation)
                print('gold:'.rjust(10), target)
                print(('sys['+str(output['source_id'])+']:').rjust(10), output['decoded'])
            #print('SYSTEM', system == target, '%.4f' % output['score'])
            #print(system)
            #print()