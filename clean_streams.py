"""Remove interleaved PAD tokens (id=0) from token streams.
Every odd position in the original files is PAD (alternating pattern).
"""
import os, numpy as np

data_dir = r'C:\Users\black\OneDrive\Desktop\fcp'
files = sorted([f for f in os.listdir(data_dir) 
                if f.startswith('token_stream_') and f.endswith('.bin') and '_clean' not in f])

for fname in files:
    path = os.path.join(data_dir, fname)
    data = np.fromfile(path, dtype=np.uint16)
    n_before = len(data)
    
    # Every other token is real (even positions), odd positions are PAD
    clean = data[0::2]
    n_after = len(clean)
    
    out_name = fname.replace('.bin', '_clean.bin')
    out_path = os.path.join(data_dir, out_name)
    clean.tofile(out_path)
    
    print(f'{fname}: {n_before:,} -> {n_after:,} tokens ({n_after/n_before*100:.1f}%)')
    print(f'  Saved to {out_name}')
