import json

with open('notebooks/colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix the print statement in cell 4
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'maturity=' in src and 'read_out=' in src:
            new_src = src.replace(
                "print(f'Collective: maturity={cfg.maturity_thresh}, read_out={cfg.collective_read_out}')",
                "print(f'Collective: maturity={cfg.collective_maturity_thresh}, read_out={cfg.collective_read_out}')"
            )
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n') + '\n'
            print('Fixed maturity_thresh reference')
            break

with open('notebooks/colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Done')
