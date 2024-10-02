import json
import numpy as np
import random

# if __name__ == "__main__":
#     root = 'data/drivelm_train.json'
#     dst_train = 'data/drivelm_train_split/drivelm_train.json'
#     dst_val = 'data/drivelm_train_split/drivelm_val.json'

#     with open(root, 'r') as f:
#         root_file = json.load(f)

#     # with open(dst_train, 'r') as f:
#     #     dst_file_train = json.load(f)

#     # with open(dst_val, 'r') as f:
#     #     dst_file_val = json.load(f)

#     num_scenes = len(root_file)
#     num_scenes_train = int(num_scenes * 0.9)
#     name_scenes = root_file.keys()
#     name_scenes_train = random.sample(name_scenes, num_scenes_train)

#     dst_train_file = {}
#     dst_val_file = {}
#     for cur_name in name_scenes:
#         if cur_name in name_scenes_train:
#             dst_train_file[cur_name] = root_file[cur_name]
#         else:
#             dst_val_file[cur_name] = root_file[cur_name]

#     print()

#     with open(dst_train, 'w') as f:
#         json.dump(dst_train_file, f, indent=4)

#     with open(dst_val, 'w') as f:
#         json.dump(dst_val_file, f, indent=4)

#     print()

import pickle
if __name__ == "__main__":
    root = "data/nuscenes/bevdetv2-nuscenes_infos_train.pkl"
    dst_train = 'data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_train_split.pkl'
    dst_val = 'data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_val_split.pkl'

    drivelm_dst_train = 'data/drivelm_train_split/drivelm_train.json'
    drivelm_dst_val = 'data/drivelm_train_split/drivelm_val.json'

    data = pickle.load(open(root, "rb"))
    data_info = data["infos"]

    # with open(dst_train, 'rb') as handle:
    #     data_info_train = pickle.load(handle)

    # with open(dst_val, 'rb') as handle:
    #     data_info_val = pickle.load(handle)

    with open(drivelm_dst_train, 'r') as f:
        drivelm_dst_file_train = json.load(f)

    with open(drivelm_dst_val, 'r') as f:
        drivelm_dst_file_val = json.load(f)

    nuscenes_file_train = {}
    nuscenes_file_val = {}

    nuscenes_file_train["infos"] = []
    nuscenes_file_val["infos"] = []

    scene_token_set = set()

    for cur_data_info in data_info:
        scene_token = cur_data_info['scene_token']
        scene_token_set.add(scene_token)
        if scene_token in drivelm_dst_file_train:
            nuscenes_file_train["infos"].append(cur_data_info)
        elif scene_token in drivelm_dst_file_val:
            nuscenes_file_val["infos"].append(cur_data_info)
        else:
            nuscenes_file_train["infos"].append(cur_data_info)

    with open(dst_train, 'wb') as handle:
        pickle.dump(nuscenes_file_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dst_val, 'wb') as handle:
        pickle.dump(nuscenes_file_val, handle, protocol=pickle.HIGHEST_PROTOCOL)


