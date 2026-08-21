import re

def extract_params(filepath):
    params = set()
    with open(filepath, 'r') as f:
        for line in f:
            m = re.match(r'\s+(\w+):\s*(?:int|float|str|bool)\s*=', line)
            if m:
                params.add(m.group(1))
    return params

big_params = extract_params('core/config.py')
mini_params = extract_params('../WideBind Mini/core/config.py')

only_mini = mini_params - big_params
only_big = big_params - mini_params

print('=== In Mini but NOT in Big (potentially missing) ===')
for p in sorted(only_mini):
    print(f'  {p}')

print('\n=== In Big but NOT in Mini ===')
for p in sorted(only_big):
    print(f'  {p}')

print(f'\nTotal: Big={len(big_params)}, Mini={len(mini_params)}')
