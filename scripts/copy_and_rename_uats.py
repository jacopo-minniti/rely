import os
import shutil

SRC_DIR = 'uats_results_v2'
DST_DIR = 'uats_results'
OFFSET = 35

for name in os.listdir(SRC_DIR):
    if name.startswith('question_'):
        try:
            idx = int(name.split('_')[1])
        except (IndexError, ValueError):
            continue
        new_idx = idx + OFFSET
        src_path = os.path.join(SRC_DIR, name)
        dst_name = f'question_{new_idx}'
        dst_path = os.path.join(DST_DIR, dst_name)
        if os.path.exists(dst_path):
            print(f'Skipping {dst_name}: already exists.')
            continue
        print(f'Copying {src_path} -> {dst_path}')
        shutil.copytree(src_path, dst_path)
