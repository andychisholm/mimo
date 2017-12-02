from mimo.model.translate import iter_decode
import ujson as json
from tqdm import tqdm
from itertools import takewhile
from collections import defaultdict

LIMIT=2048
VERBOSE=LIMIT<= 128
DECODE=True

if DECODE:
	with open('output.jsonl', 'w') as f:
		for result in tqdm(iter_decode('test.jsonl.gz', LIMIT), total=LIMIT):
			f.write(json.dumps(result) + '\n')

metrics = defaultdict(list)

results_by_entity = defaultdict(list)

with open('output.jsonl', 'r') as f:
	for line in f:
		result = json.loads(line)
		results_by_entity[result['instance_id']].append(result)

for entity_id, results in results_by_entity.items():
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

	for relation, outputs in system_relations.items():
		target = target_relations.get(relation)
		if target is not None:
			metrics[relation].append(1.0 if outputs[0]['decoded'] == target else 0.0)

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

print('Relation'.rjust(25), '|', 'Score')
print('='*50)
all_scores = []
for r, scores in metrics.items():
	all_scores.extend(scores)
	print(r.rjust(30), '|', '%.1f' % (100*sum(scores)/len(scores)))
print()
print('==========')
print('Agg:', '%.1f' % (100*sum(all_scores)/len(all_scores)))
