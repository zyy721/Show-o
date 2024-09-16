import json
import numpy as np
import random

if __name__ == "__main__":
    root = 'data/drivelm_train.json'
    dst_train = 'data/drivelm_train_split/drivelm_train.json'
    dst_val = 'data/drivelm_train_split/drivelm_val.json'

    with open(root, 'r') as f:
        root_file = json.load(f)

    with open(dst_train, 'r') as f:
        dst_file_train = json.load(f)

    with open(dst_val, 'r') as f:
        dst_file_val = json.load(f)

    num_scenes = len(root_file)
    num_scenes_train = int(num_scenes * 0.9)
    name_scenes = root_file.keys()
    name_scenes_train = random.sample(name_scenes, num_scenes_train)

    dst_train_file = {}
    dst_val_file = {}
    for cur_name in name_scenes:
        if cur_name in name_scenes_train:
            dst_train_file[cur_name] = root_file[cur_name]
        else:
            dst_val_file[cur_name] = root_file[cur_name]

    print()

    with open(dst_train, 'w') as f:
        json.dump(dst_train_file, f, indent=4)

    with open(dst_val, 'w') as f:
        json.dump(dst_val_file, f, indent=4)

    print()