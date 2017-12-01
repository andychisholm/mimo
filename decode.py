from mimo.model.translate import iter_decode
import ujson as json
from tqdm import tqdm
from itertools import takewhile
from collections import defaultdict

LIMIT=1024
VERBOSE=LIMIT<= 128
DECODE=True

if DECODE:
	with open('output.jsonl', 'w') as f:
		for result in tqdm(iter_decode('test.jsonl.gz', LIMIT), total=LIMIT):
			f.write(json.dumps(result) + '\n')

metrics = defaultdict(list)

with open('output.jsonl', 'r') as f:
	for line in f:
		result = json.loads(line)

		source = ' '.join(result['source'])
		if VERBOSE:
			print('_________________________')
			print('SOURCE')
			print(' '.join(result['source']))
			print()

		for relation, outputs in result['outputs'].items():			
			target = result['targets'].get(relation, [])
			if target:
				target = target[1:-1]
			target = ' '.join(target)
			output = sorted(outputs, key=lambda o: o['score'], reverse=True)[0]
			system = ' '.join(takewhile(lambda t: t != '</s>', output['tokens']))
			if target:
				metrics[relation].append(1.0 if system == target else 0.0)
			if VERBOSE:
				print(relation)
				print('gold:'.rjust(10), target)
				print('system:'.rjust(10), system)
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
