# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from typing import Any, Callable, Optional

import torch
from torchvision.datasets.folder import DatasetFolder, default_loader
from training.utils import image_transform

import numpy as np
import pickle

# class nuscenesDataset(DatasetFolder):
class nuscenesDataset:

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        image_size=256,
    ):
        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

        self.transform = image_transform
        self.image_size = image_size
        self.loader = loader

        # super().__init__(
        #     root,
        #     loader,
        #     IMG_EXTENSIONS if is_valid_file is None else None,
        #     transform=self.transform,
        #     target_transform=None,
        #     is_valid_file=is_valid_file,
        # )

        # with open('./training/imagenet_label_mapping', 'r') as f:
        #     self.labels = {}
        #     for l in f:
        #         num, description = l.split(":")
        #         self.labels[int(num)] = description.strip()

        # print("ImageNet dataset loaded.")

        self.images = []

        sample_rate = 2

        # imageset = "data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
        # imageset = "data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"

        imageset = root

        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())   

        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        # self.return_len = 6
        self.return_len = 3

        self.offset = 0        
        self.times = 20

        idx_image = 0
        self.idx_image_nusc_info = {}
        self.gt_ego_poses = {}
        self.gt_ego_poses_mask = {}

        self.scene_videos = []
        self.scene_videos_gt_ego_poses = []
        self.scene_videos_gt_ego_poses_mask = []

        for cur_scene_name in self.scene_names:
            self.idx_image_nusc_info[cur_scene_name] = []
            self.gt_ego_poses[cur_scene_name] = []
            self.gt_ego_poses_mask[cur_scene_name] = []

            for cur_info_idx, cur_info in enumerate(self.nusc_infos[cur_scene_name]):
                # if cur_info_idx == 0:
                #     self.gt_ego_poses[cur_scene_name].append(cur_info['gt_ego_his_trajs'][-1])

                image_path = cur_info['cams']['CAM_FRONT']['data_path']
                self.images.append(image_path)
                self.idx_image_nusc_info[cur_scene_name].append(idx_image)
                self.gt_ego_poses[cur_scene_name].append(cur_info['gt_ego_fut_trajs'][0])
                self.gt_ego_poses_mask[cur_scene_name].append(cur_info['gt_ego_fut_masks'][0])

                idx_image = idx_image + 1

            cur_scene_len = len(self.nusc_infos[cur_scene_name])
            if cur_scene_len >= self.return_len:
                for start_idx in range(cur_scene_len - (self.return_len - 1) * sample_rate):
                    # self.scene_videos.append(self.idx_image_nusc_info[cur_scene_name][start_idx:start_idx+self.return_len])
                    # self.scene_videos_gt_ego_poses.append(self.gt_ego_poses[cur_scene_name][start_idx:start_idx+self.return_len])
                    # self.scene_videos_gt_ego_poses_mask.append(self.gt_ego_poses_mask[cur_scene_name][start_idx:start_idx+self.return_len])

                    self.scene_videos.append(self.idx_image_nusc_info[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate])
                    self.scene_videos_gt_ego_poses.append(self.gt_ego_poses[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate])
                    self.scene_videos_gt_ego_poses_mask.append(self.gt_ego_poses_mask[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate])

        self.scene_videos_gt_ego_poses = np.array(self.scene_videos_gt_ego_poses)
        self.scene_videos_gt_ego_poses_mask = np.array(self.scene_videos_gt_ego_poses_mask)

        self.samples = self.images

    def __len__(self) -> int:
        # return len(self.samples)
        return len(self.scene_videos)

    def __getitem__(self, idx):

        try:
            # path, target = self.samples[idx]
            # image = self.loader(path)
            # image = self.transform(image, resolution=self.image_size)
            # input_ids = "{}".format(self.labels[target])
            # class_ids = torch.tensor(target)

            # return {'images': image, 'input_ids': input_ids, 'class_ids': class_ids}


            # path = self.samples[idx]
            # image = self.loader(path)
            # image = self.transform(image, resolution=self.image_size)


            index_list = self.scene_videos[idx]
            image_list = []
            for cur_index in index_list:
                cur_path = self.samples[cur_index]
                cur_image = self.loader(cur_path)
                cur_image = self.transform(cur_image, resolution=self.image_size)
                image_list.append(cur_image)
            image = torch.stack(image_list, dim=0)

            gt_ego_poses = torch.as_tensor(self.scene_videos_gt_ego_poses[idx])
            gt_ego_poses_mask = torch.as_tensor(self.scene_videos_gt_ego_poses_mask[idx])

            return {'images': image, 'gt_ego_poses': gt_ego_poses, 'gt_ego_poses_mask': gt_ego_poses_mask}

        except Exception as e:
            print(e)
            return self.__getitem__(idx+1)

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('input_ids'):
                batched[k] = torch.stack(v, dim=0)

        return batched

# class nuscenesDataset(DatasetFolder):
class nuscenesDatasetVal:

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        image_size=256,
    ):
        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

        self.transform = image_transform
        self.image_size = image_size
        self.loader = loader

        # super().__init__(
        #     root,
        #     loader,
        #     IMG_EXTENSIONS if is_valid_file is None else None,
        #     transform=self.transform,
        #     target_transform=None,
        #     is_valid_file=is_valid_file,
        # )

        # with open('./training/imagenet_label_mapping', 'r') as f:
        #     self.labels = {}
        #     for l in f:
        #         num, description = l.split(":")
        #         self.labels[int(num)] = description.strip()

        # print("ImageNet dataset loaded.")

        self.images = []

        sample_rate = 2

        # imageset = "data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
        # imageset = "data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"

        imageset = root

        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())   

        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        # self.return_len = 6
        self.return_len = 3

        self.offset = 0        
        self.times = 20

        idx_image = 0
        self.idx_image_nusc_info = {}
        self.gt_ego_poses = {}
        self.gt_ego_poses_mask = {}

        self.scene_videos = []
        self.scene_videos_gt_ego_poses = []
        self.scene_videos_gt_ego_poses_mask = []

        for cur_scene_name in self.scene_names:
            self.idx_image_nusc_info[cur_scene_name] = []
            self.gt_ego_poses[cur_scene_name] = []
            self.gt_ego_poses_mask[cur_scene_name] = []

            for cur_info_idx, cur_info in enumerate(self.nusc_infos[cur_scene_name]):
                # if cur_info_idx == 0:
                #     self.gt_ego_poses[cur_scene_name].append(cur_info['gt_ego_his_trajs'][-1])

                image_path = cur_info['cams']['CAM_FRONT']['data_path']
                self.images.append(image_path)
                self.idx_image_nusc_info[cur_scene_name].append(idx_image)
                self.gt_ego_poses[cur_scene_name].append(cur_info['gt_ego_fut_trajs'][0])
                self.gt_ego_poses_mask[cur_scene_name].append(cur_info['gt_ego_fut_masks'][0])

                idx_image = idx_image + 1

            cur_scene_len = len(self.nusc_infos[cur_scene_name])
            if cur_scene_len >= self.return_len:
                for start_idx in range(cur_scene_len - (self.return_len - 1) * sample_rate):
                    # self.scene_videos.append(self.idx_image_nusc_info[cur_scene_name][start_idx:start_idx+self.return_len])
                    # self.scene_videos_gt_ego_poses.append(self.gt_ego_poses[cur_scene_name][start_idx:start_idx+self.return_len])
                    # self.scene_videos_gt_ego_poses_mask.append(self.gt_ego_poses_mask[cur_scene_name][start_idx:start_idx+self.return_len])

                    self.scene_videos.append(self.idx_image_nusc_info[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate])
                    self.scene_videos_gt_ego_poses.append(self.gt_ego_poses[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate])
                    self.scene_videos_gt_ego_poses_mask.append(self.gt_ego_poses_mask[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate])

        self.scene_videos_gt_ego_poses = np.array(self.scene_videos_gt_ego_poses)
        self.scene_videos_gt_ego_poses_mask = np.array(self.scene_videos_gt_ego_poses_mask)

        self.samples = self.images

    def __len__(self) -> int:
        # return len(self.samples)
        return len(self.scene_videos)

    def __getitem__(self, idx):

        try:
            # path, target = self.samples[idx]
            # image = self.loader(path)
            # image = self.transform(image, resolution=self.image_size)
            # input_ids = "{}".format(self.labels[target])
            # class_ids = torch.tensor(target)

            # return {'images': image, 'input_ids': input_ids, 'class_ids': class_ids}


            # path = self.samples[idx]
            # image = self.loader(path)
            # image = self.transform(image, resolution=self.image_size)


            index_list = self.scene_videos[idx]
            image_list, path_list = [], []
            for cur_index in index_list:
                cur_path = self.samples[cur_index]
                cur_image = self.loader(cur_path)
                cur_image = self.transform(cur_image, resolution=self.image_size)
                image_list.append(cur_image)
                path_list.append(cur_path)
            image = torch.stack(image_list, dim=0)

            # gt_ego_poses = torch.as_tensor(self.scene_videos_gt_ego_poses[idx])
            # gt_ego_poses_mask = torch.as_tensor(self.scene_videos_gt_ego_poses_mask[idx])

            # return {'images': image, 'gt_ego_poses': gt_ego_poses, 'gt_ego_poses_mask': gt_ego_poses_mask}
            return {'images': image, 'path': path_list}

        except Exception as e:
            print(e)
            return self.__getitem__(idx+1)

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('input_ids'):
                batched[k] = torch.stack(v, dim=0)

        return batched


if __name__ == '__main__':
    pass
