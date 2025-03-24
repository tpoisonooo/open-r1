from datasets import load_dataset
import sys
import random

ds = load_dataset(sys.argv[1])

import pdb
train_data = ds['train']
print(len(train_data))
print('-' * 79)
# indexes = random.sample(range(len(train_data)), 100)
indexes = range(len(train_data))
print_data = train_data.select(indexes)

for data in print_data:
    if data['source_type'] == 'math':
        pdb.set_trace()

        if '```python' in str(data['gold_standard_solution']):

            pass

    print(data.keys())
print('-' * 79)