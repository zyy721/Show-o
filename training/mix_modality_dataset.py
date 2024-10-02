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

import copy
import json
import os
from functools import partial

import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from training.utils import image_transform
from llava.llava import conversation as conversation_lib

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."

import pickle
import numpy as np
import collections
from transformers import CLIPImageProcessor


def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_v0(
        sources,
        tokenizer,
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )


# class nuscenesDataset(DatasetFolder):
class MixModalityDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        # root: str,
        # loader: Callable[[str], Any] = default_loader,
        # is_valid_file: Optional[Callable[[str], bool]] = None,
        image_size=256,
    ):
        super(MixModalityDataset, self).__init__()

        self.tokenizer = tokenizer

        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

        self.transform = image_transform
        self.image_size = image_size
        # self.loader = loader

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

        self.answers = []
        self.questions = []
        self.images = []
        self.tmp_imglist = []

        # sample_rate = 2

        # imageset = "data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
        # imageset = "data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"

        # self.default_video()
        self.default_drivelm()


    def default_video(self):
        self.images = []

        imageset = "data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
        # imageset = "data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"

        # imageset = root

        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())   

        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.return_len = 6
        # self.return_len = 3
        # self.return_len = 4

        past_length = 3
        future_length = 3
        sample_rate = 2

        self.offset = 0        
        self.times = 20

        idx_image = 0
        self.idx_image_nusc_info = {}
        self.gt_ego_poses = {}
        self.gt_ego_poses_mask = {}
        self.time_stamp = {}
        self.token = {}

        self.scene_videos = []
        self.scene_videos_gt_ego_poses = []
        self.scene_videos_gt_ego_poses_mask = []

        self.dict_scene_videos = {}
        self.dict_scene_videos_gt_ego_poses = {}
        self.dict_scene_videos_gt_ego_poses_mask = {}
        self.dict_scene_token = {}

        for cur_scene_name in self.scene_names:
            self.idx_image_nusc_info[cur_scene_name] = []
            self.gt_ego_poses[cur_scene_name] = []
            self.gt_ego_poses_mask[cur_scene_name] = []
            self.time_stamp[cur_scene_name] = []
            self.token[cur_scene_name] = []

            for cur_info_idx, cur_info in enumerate(self.nusc_infos[cur_scene_name]):
                # if cur_info_idx == 0:
                #     self.gt_ego_poses[cur_scene_name].append(cur_info['gt_ego_his_trajs'][-1])

                image_path = cur_info['cams']['CAM_FRONT']['data_path']
                self.images.append(image_path)
                self.idx_image_nusc_info[cur_scene_name].append(idx_image)
                self.gt_ego_poses[cur_scene_name].append(cur_info['gt_ego_fut_trajs'][0])
                self.gt_ego_poses_mask[cur_scene_name].append(cur_info['gt_ego_fut_masks'][0])
                self.time_stamp[cur_scene_name].append(cur_info['timestamp'])
                self.token[cur_scene_name].append(cur_info['token'])

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

                    cur_token = self.token[cur_scene_name][start_idx]
                    cur_time_stamp = self.time_stamp[cur_scene_name][start_idx]
                    self.dict_scene_videos[cur_time_stamp] = self.idx_image_nusc_info[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate]
                    self.dict_scene_videos_gt_ego_poses[cur_time_stamp] = self.gt_ego_poses[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate]
                    self.dict_scene_videos_gt_ego_poses_mask[cur_time_stamp] = self.gt_ego_poses_mask[cur_scene_name][start_idx:start_idx+self.return_len*sample_rate:sample_rate]
                    self.dict_scene_token[cur_token] = cur_time_stamp

        # self.scene_videos_gt_ego_poses = np.array(self.scene_videos_gt_ego_poses)
        # self.scene_videos_gt_ego_poses_mask = np.array(self.scene_videos_gt_ego_poses_mask)

        self.samples = self.images


    # def default_drivelm(self, ann_paths):
    def default_drivelm(self):
        self.temporal_length = 6
        past_length = 3
        future_length = 3
        sample_rate = 2

        # dict_nusc_info = {}
        # for cur_scene_name in self.scene_names:
        #     for cur_info_idx, cur_nusc_info in enumerate(self.nusc_infos[cur_scene_name]):            
        #         scene_token = cur_nusc_info['token']
        #         dict_nusc_info[scene_token] = cur_nusc_info

        # # self.annotation = json.load(open(ann_paths[0], "r"))
        # # self.annotation = json.load(open('data/drivelm_train.json', "r"))
        # self.annotation = json.load(open('data/drivelm_train_split/drivelm_train.json', "r"))
        # bevdetv2_nuscenes_infos_train = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_train.pkl", "rb"))
        # self.data_info = bevdetv2_nuscenes_infos_train["infos"]
        # for idx, info in enumerate(self.data_info):
        #     scene_token = info['token']
        #     timestamp = info['cams']['CAM_FRONT']['timestamp']
        #     image_path = info['cams']["CAM_FRONT"]['data_path']
        #     if scene_token in dict_nusc_info:
        #         info_2 = dict_nusc_info[scene_token]
        #         print()

        self.annotation = json.load(open('data/drivelm_train.json', "r"))
        self.data_info = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_train_split.pkl', "rb"))["infos"]

        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']
            # temporal data only for image path
            past_tmp_image = []
            for tmp in range(0, past_length*sample_rate, sample_rate):
                if scene_token != self.data_info[idx-tmp]['scene_token']:
                    continue
                tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                past_tmp_image.append(tmp_path)

            # if the image path is not equal self.temporal length, then use the duplicate image path
            past_tmp_image = past_tmp_image[::-1]
            if len(past_tmp_image) != past_length:
                past_tmp_image = ['pad'] * (past_length - len(past_tmp_image)) + past_tmp_image
            assert len(past_tmp_image) == past_length

            # temporal data only for image path
            future_tmp_image = []
            for tmp in range(sample_rate, (future_length+1)*sample_rate, sample_rate):
                if idx+tmp > len(self.data_info)-1 or scene_token != self.data_info[idx+tmp]['scene_token']:
                    continue
                tmp_path = self.data_info[idx+tmp]['cams']['CAM_FRONT']['data_path']
                future_tmp_image.append(tmp_path)

            # if the image path is not equal self.temporal length, then use the duplicate image path
            if len(future_tmp_image) != future_length:
                future_tmp_image = future_tmp_image + ['pad'] * (future_length - len(future_tmp_image))
            assert len(future_tmp_image) == future_length

            tmp_image = past_tmp_image + future_tmp_image

            if scene_token not in self.annotation:
                self.questions.append('pad')
                self.answers.append(['pad'])
                self.tmp_imglist.append(tmp_image)
            else:
                value = self.annotation[scene_token]
                # scene_description = value['scene_description']
                scene_key_frame = value['key_frame']
                frame_id = str(timestamp)
                if frame_id in scene_key_frame:
                    value1 = scene_key_frame[frame_id]

                    if "Perception" in value1:
                        Perception_q = value1['Perception']['q']
                        Perception_a = value1['Perception']['a']
                    else:
                        Perception_q = []
                        Perception_a = []

                    if "Prediction and Planning" in value1:
                        Prediction_q = value1['Prediction and Planning']['q']
                        Prediction_a = value1['Prediction and Planning']['a']
                    else:
                        Prediction_q = []
                        Prediction_a = []
                                        

                    Question = Perception_q + Prediction_q
                    Answer = Perception_a + Prediction_a

                
                    assert len(Question) == len(Answer)

                    for idx in range(len(Question)):    
                        cur_question = Question[idx]
                        if "CAM_FRONT_LEFT" in cur_question or "CAM_FRONT_RIGHT" in cur_question \
                        or "CAM_BACK" in cur_question or "CAM_BACK_LEFT" in cur_question or "CAM_BACK_LEFT" in cur_question:
                            continue
                        if "front left" in cur_question or "front right" in cur_question \
                        or "back" in cur_question:
                            continue

                        cur_answer = Answer[idx]
                        if "CAM_FRONT_LEFT" in cur_answer or "CAM_FRONT_RIGHT" in cur_answer \
                        or "CAM_BACK" in cur_answer or "CAM_BACK_LEFT" in cur_answer or "CAM_BACK_LEFT" in cur_answer:
                            continue
                        if "front left" in cur_answer or "front right" in cur_answer \
                        or "back" in cur_answer:
                            continue

                        # if "CAM_FRONT" in cur_question or "CAM_FRONT" in cur_answer:
                        #     continue

                        self.questions.append(Question[idx])
                        self.answers.append([Answer[idx]])
                        self.tmp_imglist.append(tmp_image)

                else:
                    self.questions.append('pad')
                    self.answers.append(['pad'])
                    self.tmp_imglist.append(tmp_image)

        # print()

    def __len__(self) -> int:
        # return len(self.samples)
        # return len(self.scene_videos)
        return len(self.questions)

    def __getitem__(self, i):
        cur_question = self.questions[i]
        cur_answer = self.answers[i][0]

        cur_question = cur_question[3:]
        cur_answer = cur_answer[3:]

        sources = {
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + cur_question
                },
                {
                    "from": "gpt",
                    "value": cur_answer
                },
            ]
        }
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # cur_image_path = self.images[i]
        # try:
        #     image = Image.open(cur_image_path).convert('RGB')
        #     image = image_transform(image)
        # except:
        #     print("Read image error. Use dummy data.")
        #     crop_size = 256
        #     image = torch.zeros(3, crop_size, crop_size)

        path_list = self.tmp_imglist[i]
        image_list = []
        for cur_image_path in path_list:
            cur_image = Image.open(cur_image_path).convert('RGB')
            cur_image = image_transform(cur_image)
            image_list.append(cur_image)
        image = torch.stack(image_list, dim=0)

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        data_dict = preprocess_v0(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        # image exist in the data
        # if 'image' in self.list_data_dict[i]:
        #     data_dict['image'] = image
        # else:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = 256
        #     data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        data_dict['image'] = image

        return data_dict


class VideoPredMixModalityDataset(Dataset):

    def __init__(
        self,
        tokenizer=None,
        image_size=256,
    ):
        super(VideoPredMixModalityDataset, self).__init__()

        self.tokenizer = tokenizer
        self.transform = image_transform
        self.image_size = image_size

        # self.answers = []
        # self.questions = []
        # self.images = []
        self.tmp_imglist = []

        vision_tower_name = "openai/clip-vit-large-patch14-336"
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        self.default_drivelm()


    def default_drivelm(self):
        self.temporal_length = 6
        past_length = 3
        future_length = 3
        sample_rate = 2

        self.annotation = json.load(open('data/drivelm_train.json', "r"))
        self.data_info = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_train_split.pkl', "rb"))["infos"]

        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']
            # temporal data only for image path
            past_tmp_image = []
            for tmp in range(0, past_length*sample_rate, sample_rate):
                if scene_token != self.data_info[idx-tmp]['scene_token']:
                    continue
                tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                past_tmp_image.append(tmp_path)

            # if the image path is not equal self.temporal length, then use the duplicate image path
            past_tmp_image = past_tmp_image[::-1]
            if len(past_tmp_image) != past_length:
                past_tmp_image = ['pad'] * (past_length - len(past_tmp_image)) + past_tmp_image
            assert len(past_tmp_image) == past_length

            # temporal data only for image path
            future_tmp_image = []
            for tmp in range(sample_rate, (future_length+1)*sample_rate, sample_rate):
                if idx+tmp > len(self.data_info)-1 or scene_token != self.data_info[idx+tmp]['scene_token']:
                    continue
                tmp_path = self.data_info[idx+tmp]['cams']['CAM_FRONT']['data_path']
                future_tmp_image.append(tmp_path)

            if len(future_tmp_image) > 0:
                # if the image path is not equal self.temporal length, then use the duplicate image path
                if len(future_tmp_image) != future_length:
                    future_tmp_image = future_tmp_image + ['pad'] * (future_length - len(future_tmp_image))
                assert len(future_tmp_image) == future_length

                tmp_image = past_tmp_image + future_tmp_image

                # self.questions.append('pad')
                # self.answers.append(['pad'])
                self.tmp_imglist.append(tmp_image)

        # print()

    def __len__(self) -> int:
        # return len(self.questions)
        return len(self.tmp_imglist)
    

    def __getitem__(self, i):
        # cur_question = self.questions[i]
        # cur_answer = self.answers[i][0]

        # cur_question = cur_question[3:]
        # cur_answer = cur_answer[3:]

        # sources = {
        #     "conversations": [
        #         {
        #             "from": "human",
        #             "value": "<image>\n" + cur_question
        #         },
        #         {
        #             "from": "gpt",
        #             "value": cur_answer
        #         },
        #     ]
        # }
        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # cur_image_path = self.images[i]
        # try:
        #     image = Image.open(cur_image_path).convert('RGB')
        #     image = image_transform(image)
        # except:
        #     print("Read image error. Use dummy data.")
        #     crop_size = 256
        #     image = torch.zeros(3, crop_size, crop_size)

        path_list = self.tmp_imglist[i]
        image_list = []
        bool_pad_image_list = []
        for cur_image_path in path_list:
            if cur_image_path == 'pad':
                crop_size = 256
                cur_image = torch.zeros(3, crop_size, crop_size)
                bool_pad_image_list.append(True)
            else:
                cur_image = Image.open(cur_image_path).convert('RGB')

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                cur_image = expand2square(cur_image, tuple(int(x*255) for x in self.clip_image_processor.image_mean))

                cur_image = image_transform(cur_image)
                bool_pad_image_list.append(False)
            image_list.append(cur_image)
        image = torch.stack(image_list, dim=0)
        bool_pad_image = torch.tensor(bool_pad_image_list)
        data_dict = {}
        data_dict['bool_pad_image'] = bool_pad_image

        # sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        # data_dict = preprocess_v0(sources, self.tokenizer)

        # if isinstance(i, int):
        #     data_dict = dict(input_ids=data_dict["input_ids"][0],
        #                      labels=data_dict["labels"][0],
        #                      input_ids_system=data_dict["input_ids_system"][0])

        # image exist in the data
        # if 'image' in self.list_data_dict[i]:
        #     data_dict['image'] = image
        # else:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = 256
        #     data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        data_dict['images'] = image

        return data_dict

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('input_ids'):
                batched[k] = torch.stack(v, dim=0)

        return batched


class DriveLMMixModalityDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        image_size=256,
    ):
        super(DriveLMMixModalityDataset, self).__init__()

        self.tokenizer = tokenizer
        self.transform = image_transform
        self.image_size = image_size

        self.answers = []
        self.questions = []
        self.images = []
        self.tmp_imglist = []

        vision_tower_name = "openai/clip-vit-large-patch14-336"
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        self.default_drivelm()


    def default_drivelm(self):
        self.temporal_length = 6
        past_length = 3
        future_length = 3
        sample_rate = 2

        # self.annotation = json.load(open('data/drivelm_train.json', "r"))
        self.annotation = json.load(open('data/converted_drivelm_train_range_zero_one.json', "r"))
        self.data_info = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_train_split.pkl', "rb"))["infos"]

        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']
            # temporal data only for image path
            past_tmp_image = []
            for tmp in range(0, past_length*sample_rate, sample_rate):
                if scene_token != self.data_info[idx-tmp]['scene_token']:
                    continue
                tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                past_tmp_image.append(tmp_path)

            # if the image path is not equal self.temporal length, then use the duplicate image path
            past_tmp_image = past_tmp_image[::-1]
            if len(past_tmp_image) != past_length:
                past_tmp_image = ['pad'] * (past_length - len(past_tmp_image)) + past_tmp_image
            assert len(past_tmp_image) == past_length

            # temporal data only for image path
            future_tmp_image = []
            for tmp in range(sample_rate, (future_length+1)*sample_rate, sample_rate):
                if idx+tmp > len(self.data_info)-1 or scene_token != self.data_info[idx+tmp]['scene_token']:
                    continue
                tmp_path = self.data_info[idx+tmp]['cams']['CAM_FRONT']['data_path']
                future_tmp_image.append(tmp_path)

            # if the image path is not equal self.temporal length, then use the duplicate image path
            if len(future_tmp_image) != future_length:
                future_tmp_image = future_tmp_image + ['pad'] * (future_length - len(future_tmp_image))
            assert len(future_tmp_image) == future_length

            tmp_image = past_tmp_image + future_tmp_image

            if scene_token in self.annotation:
                value = self.annotation[scene_token]
                # scene_description = value['scene_description']
                scene_key_frame = value['key_frame']
                frame_id = str(timestamp)
                if frame_id in scene_key_frame:
                    value1 = scene_key_frame[frame_id]

                    if "Perception" in value1:
                        Perception_q = value1['Perception']['q']
                        Perception_a = value1['Perception']['a']
                    else:
                        Perception_q = []
                        Perception_a = []

                    if "Prediction and Planning" in value1:
                        Prediction_q = value1['Prediction and Planning']['q']
                        Prediction_a = value1['Prediction and Planning']['a']
                    else:
                        Prediction_q = []
                        Prediction_a = []
                                        

                    Question = Perception_q + Prediction_q
                    Answer = Perception_a + Prediction_a

                
                    assert len(Question) == len(Answer)

                    for idx in range(len(Question)):    
                        cur_question = Question[idx]
                        if "CAM_FRONT_LEFT" in cur_question or "CAM_FRONT_RIGHT" in cur_question \
                        or "CAM_BACK" in cur_question or "CAM_BACK_LEFT" in cur_question or "CAM_BACK_LEFT" in cur_question:
                            continue
                        if "front left" in cur_question or "front right" in cur_question \
                        or "back" in cur_question:
                            continue

                        cur_answer = Answer[idx]
                        if "CAM_FRONT_LEFT" in cur_answer or "CAM_FRONT_RIGHT" in cur_answer \
                        or "CAM_BACK" in cur_answer or "CAM_BACK_LEFT" in cur_answer or "CAM_BACK_LEFT" in cur_answer:
                            continue
                        if "front left" in cur_answer or "front right" in cur_answer \
                        or "back" in cur_answer:
                            continue

                        # if "CAM_FRONT" in cur_question or "CAM_FRONT" in cur_answer:
                        #     continue

                        self.questions.append(Question[idx])
                        self.answers.append([Answer[idx]])
                        self.tmp_imglist.append(tmp_image)

        # print()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, i):
        cur_question = self.questions[i]
        cur_answer = self.answers[i][0]

        cur_question = cur_question[3:]
        cur_answer = cur_answer[3:]

        sources = {
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + cur_question
                },
                {
                    "from": "gpt",
                    "value": cur_answer
                },
            ]
        }
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # cur_image_path = self.images[i]
        # try:
        #     image = Image.open(cur_image_path).convert('RGB')
        #     image = image_transform(image)
        # except:
        #     print("Read image error. Use dummy data.")
        #     crop_size = 256
        #     image = torch.zeros(3, crop_size, crop_size)


        path_list = self.tmp_imglist[i]
        image_list = []
        bool_pad_image_list = []
        for cur_image_path in path_list:
            if cur_image_path == 'pad':
                crop_size = 256
                cur_image = torch.zeros(3, crop_size, crop_size)
                bool_pad_image_list.append(True)
            else:
                cur_image = Image.open(cur_image_path).convert('RGB')

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                cur_image = expand2square(cur_image, tuple(int(x*255) for x in self.clip_image_processor.image_mean))

                cur_image = image_transform(cur_image, save=True)
                bool_pad_image_list.append(False)
            image_list.append(cur_image)
        image = torch.stack(image_list, dim=0)
        bool_pad_image = torch.tensor(bool_pad_image_list)
        

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        data_dict = preprocess_v0(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        # image exist in the data
        # if 'image' in self.list_data_dict[i]:
        #     data_dict['image'] = image
        # else:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = 256
        #     data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        data_dict['image'] = image
        data_dict['bool_pad_image'] = bool_pad_image

        return data_dict



def collate_fn(
        instances,
        tokenizer=None,
        max_length=77,
):
    input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "input_ids_system"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    input_ids_system = torch.stack(input_ids_system, dim=0)

    offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]

    if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
        pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)

        pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)

    min_max_len = min(
        max_length - input_ids_system.shape[-1],
        tokenizer.model_max_length - input_ids_system.shape[-1],
    )

    input_ids = input_ids[:, :min_max_len]
    labels = labels[:, :min_max_len]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        input_ids_system=input_ids_system,
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    bool_pad_image = [instance['bool_pad_image'] for instance in instances]
    batch['bool_pad_image'] = torch.stack(bool_pad_image)

    return batch


def get_drivelm_mix_modality_data_loader(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        phase,
):
    # train_dataset = MixModalityDataset(
    #     tokenizer,
    #     # phase,
    # )
    train_dataset = DriveLMMixModalityDataset(
        tokenizer,
        # phase,
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader, len(train_dataset)


def get_mix_modality_data_loader(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        phase,
):
    train_dataset = MixModalityDataset(
        tokenizer,
        # phase,
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader, len(train_dataset)


if __name__ == '__main__':
    pass
