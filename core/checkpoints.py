"""Checkpoint discovery helpers."""
import glob
import os


def find_latest_ckpt(save_dir):
    """Return path to the freshest checkpoint in save_dir.

    Priority: interrupt_step_* > step_* > best.pt. Within a group the
    highest step wins (sorted by name). Falls back to the newest *.pt by
    mtime, or None if no checkpoints exist.
    """
    groups = [
        glob.glob(os.path.join(save_dir, 'interrupt_step_*.pt')),
        glob.glob(os.path.join(save_dir, 'step_*.pt')),
        glob.glob(os.path.join(save_dir, 'best.pt')),
    ]
    for group in groups:
        if group:
            return sorted(group)[-1]
    all_pt = glob.glob(os.path.join(save_dir, '*.pt'))
    if all_pt:
        return max(all_pt, key=os.path.getmtime)
    return None
