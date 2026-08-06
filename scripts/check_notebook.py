import json

with open('notebooks/colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Cells:', len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'Build Model' in src:
            print(f'Cell {i}: Build Model')
            for line in cell['source']:
                if 'cfg.' in line or 'head_mode' in line or 'bind_twist' in line:
                    print('  ', line.strip())
        if 'pw_diag' in src:
            print(f'Cell {i}: STILL HAS CODEC METRICS!')
        if 'head_temp' in src:
            print(f'Cell {i}: Has new head_temp metric')
