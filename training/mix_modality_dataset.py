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
from PIL import Image, ImageDraw
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
import re


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


class MixModalityDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        image_size=256,
        data_info=None,
        world_size=None,
        local_rank=None,
    ):
        super(MixModalityDataset, self).__init__()

        self.tokenizer = tokenizer
        self.transform = image_transform
        self.image_size = image_size

        self.answers = []
        self.questions = []
        self.images = []
        self.tmp_imglist = []
        self.tasks = []

        vision_tower_name = "openai/clip-vit-large-patch14-336"
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        if data_info is None:
            self.data_info = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_val_split.pkl', "rb"))["infos"]
        else:
            self.data_info = data_info

        self.default_drivelm()

        if local_rank is not None:
            num_samples = len(self.questions)
            each_samples = num_samples // world_size

            if local_rank == (world_size - 1):
                self.questions = self.questions[each_samples*local_rank:]
                self.answers = self.answers[each_samples*local_rank:]
                self.tmp_imglist = self.tmp_imglist[each_samples*local_rank:]
                self.tasks = self.tasks[each_samples*local_rank:]
            else:
                self.questions = self.questions[each_samples*local_rank:each_samples*(local_rank+1)]
                self.answers = self.answers[each_samples*local_rank:each_samples*(local_rank+1)]
                self.tmp_imglist = self.tmp_imglist[each_samples*local_rank:each_samples*(local_rank+1)]
                self.tasks = self.tasks[each_samples*local_rank:each_samples*(local_rank+1)]


    def default_drivelm(self):
        self.temporal_length = 6
        # past_length = 3
        # future_length = 3
        past_length = 2
        future_length = 2
        sample_rate = 2

        # self.annotation = json.load(open('data/drivelm_train.json', "r"))
        # self.data_info = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_train_split.pkl', "rb"))["infos"]

        self.annotation = json.load(open('data/converted_drivelm_train_range_zero_one.json', "r"))
        # self.data_info = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train_split/bevdetv2-nuscenes_infos_val_split.pkl', "rb"))["infos"]

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

            # if len(future_tmp_image) > 0:
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
                        Perception_task = ['Perception'] * len(Perception_q)
                    else:
                        Perception_q = []
                        Perception_a = []
                        Perception_task = []

                    if "Prediction and Planning" in value1:
                        Prediction_q = value1['Prediction and Planning']['q']
                        Prediction_a = value1['Prediction and Planning']['a']
                        Prediction_task = ['Prediction and Planning'] * len(Prediction_q)
                    else:
                        Prediction_q = []
                        Prediction_a = []
                        Prediction_task = []
       

                    Question = Perception_q + Prediction_q
                    Answer = Perception_a + Prediction_a
                    Name_task = Perception_task + Prediction_task
                
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


                        self.questions.append(Question[idx])
                        self.answers.append([Answer[idx]])
                        self.tmp_imglist.append(tmp_image)
                        self.tasks.append(Name_task[idx])

        # all_idx = []
        # for idx_question, cur_question in enumerate(self.questions):
        #     # if "What is the movement of object <c1,CAM_FRONT,0.47,0.55>?" in cur_question:
        #     # if "Would <c2,CAM_FRONT,0.20,0.59> be in the moving direction of the ego vehicle?" in cur_question:
        #     # if "What is the moving status of object <c3,CAM_FRONT,0.76,0.52>?" in cur_question:
        #     # if "What is the future state of <c1,CAM_FRONT,0.61,0.59>?" in cur_question:
        #     # if "What is the goal action of the ego vehicle?" in cur_question:
        #     # if "What is the visual description of <c2,CAM_FRONT,0.01,0.52>?" in cur_question:
        #     # if "Would <c2,CAM_FRONT,0.17,0.59> be in the moving direction of the ego vehicle?" in cur_question:
        #     # if "What actions taken by the ego vehicle can lead to a collision with <c2,CAM_FRONT,0.49,0.53>?" in cur_question:
        #     # if "Would <c4,CAM_FRONT,0.81,0.55> be in the moving direction of the ego vehicle?" in cur_question:
        #     # if "What is the movement of object <c2,CAM_FRONT,0.98,0.53>?" in cur_question:
        #     # if "What is the movement of object <c1,CAM_FRONT,0.01,0.54>?" in cur_question:
        #     # if "What is the movement of object <c3,CAM_FRONT,0.97,0.53>?" in cur_question:
        #     # if "What is the movement of object <c1,CAM_FRONT,0.37,0.53>?" in cur_question:
        #     # if "What is the movement of object <c5,CAM_FRONT,0.93,0.56>?" in cur_question:
        #     # if "What is the movement of object <c3,CAM_FRONT,0.16,0.58>?" in cur_question:
        #     # if "What is the movement of object <c2,CAM_FRONT,0.83,0.58>?" in cur_question:
        #     # if "What is the movement of object <c2,CAM_FRONT,0.62,0.53>?" in cur_question:
        #     # if "What is the movement of object <c1,CAM_FRONT,0.64,0.52>?" in cur_question:
        #     # if "What is the movement of object <c4,CAM_FRONT,0.53,0.50>?" in cur_question:
        #     if "Is it necessary for the ego vehicle to take <c4,CAM_FRONT,0.74,0.43> into account?" in cur_question:
        #         all_idx.append(idx_question)

        # print()

    def __len__(self) -> int:
        # return len(self.questions)
        return len(self.tmp_imglist)
    

    def __getitem__(self, i):
        # i = 1710
        # i = 1722
        # i = 1729
        # i = 4944
        # i = 12720
        # i = 1677
        # i = 8333
        # i = 8234
        # i = 3248
        # i = 3238
        # i = 3193
        # i = 1618
        # i = 3141
        # i = 13308
        # i = 13298
        # i = 3121
        # i = 5353
        # i = 8616
        i = 7465


        cur_question = self.questions[i]
        cur_answer = self.answers[i][0]

        cur_question = cur_question[3:]
        cur_answer = cur_answer[3:]

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
        for cur_timestamp, cur_image_path in enumerate(path_list):
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
                

                cur_image_copy = cur_image.copy()
                resolution = 256
                from torchvision import transforms
                cur_image_copy = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(cur_image_copy)
                cur_image_copy = transforms.CenterCrop((resolution, resolution))(cur_image_copy)
                if cur_timestamp == 1:
                    def convert_string(input_string):
                        pattern = r"<([^>]+),(\d+\.\d+),(\d+\.\d+)>"
                        match = re.search(pattern, input_string)
                        string_value, float_value1, float_value2 = match.group(1, 2, 3)
                        float_value1 = float(float_value1)
                        float_value2 = float(float_value2)
                        return float_value1, float_value2

                    x_range_0_1, y_range_0_1 = convert_string(cur_question)
                    point_coordinates = (int(cur_image_copy.size[0]*x_range_0_1), int(cur_image_copy.size[1]*y_range_0_1))  # x, y coordinates
                    draw = ImageDraw.Draw(cur_image_copy)
                    point_color = (255, 0, 0)  # Red color
                    point_size = 5  # Size of the point
                    draw.rectangle([point_coordinates[0] - point_size, point_coordinates[1] - point_size,
                                    point_coordinates[0] + point_size, point_coordinates[1] + point_size], 
                                fill=point_color)
                cur_image_copy.save('orig_{}.jpg'.format(cur_timestamp))


                cur_image = image_transform(cur_image)

                bool_pad_image_list.append(False)
            image_list.append(cur_image)
        image = torch.stack(image_list, dim=0)
        bool_pad_image = torch.tensor(bool_pad_image_list)
        data_dict = {}
        data_dict['bool_pad_image'] = bool_pad_image

        # TODO
        input_ids = self.tokenizer(['USER: \n' + cur_question + ' ASSISTANT:'])[
            'input_ids']
        input_ids = torch.tensor(input_ids[0])
        data_dict['input_ids'] = input_ids

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
        data_dict['question'] = cur_question
        data_dict['answer'] = cur_answer
        data_dict['task'] = self.tasks[i]

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
        w_clip=False
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
        self.w_clip = w_clip

        self.default_drivelm()


    def default_drivelm(self):
        # self.temporal_length = 6
        # past_length = 3
        # future_length = 3
        self.temporal_length = 4
        self.past_length = 2
        future_length = 2
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
            for tmp in range(0, self.past_length*sample_rate, sample_rate):
                if scene_token != self.data_info[idx-tmp]['scene_token']:
                    continue
                tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                past_tmp_image.append(tmp_path)

            # if the image path is not equal self.temporal length, then use the duplicate image path
            past_tmp_image = past_tmp_image[::-1]
            if len(past_tmp_image) != self.past_length:
                past_tmp_image = ['pad'] * (self.past_length - len(past_tmp_image)) + past_tmp_image
            assert len(past_tmp_image) == self.past_length

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
                    "value": "<image>\nGiven <frame {}> as the current frame. ".format(self.past_length-1) + cur_question
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
            if self.w_clip:
                if cur_image_path == 'pad':
                    crop_size = 336
                    cur_image = torch.zeros(3, crop_size, crop_size)
                    bool_pad_image_list.append(True)
                else:
                    cur_image = Image.open(cur_image_path).convert('RGB')
                    cur_image = self.clip_image_processor.preprocess(cur_image, return_tensors='pt')['pixel_values'][0]
                    bool_pad_image_list.append(False)
            else:
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
        w_clip=False,
):
    # train_dataset = MixModalityDataset(
    #     tokenizer,
    #     # phase,
    # )
    train_dataset = DriveLMMixModalityDataset(
        tokenizer,
        # phase,
        w_clip=w_clip,
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


def collate_fn_val(
        instances,
        tokenizer=None,
        max_length=77,
):
    # input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
    #                                             for key in ("input_ids", "labels", "input_ids_system"))
    input_ids = [instance['input_ids'] for instance in instances]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    # labels = torch.nn.utils.rnn.pad_sequence(labels,
    #                                          batch_first=True,
    #                                          padding_value=IGNORE_INDEX)
    # input_ids_system = torch.stack(input_ids_system, dim=0)

    # offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]

    # if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
    #     pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
    #     input_ids = torch.cat([input_ids, pad_tube], dim=1)

    #     pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
    #     labels = torch.cat([labels, pad_tube], dim=1)

    # min_max_len = min(
    #     max_length - input_ids_system.shape[-1],
    #     tokenizer.model_max_length - input_ids_system.shape[-1],
    # )

    # input_ids = input_ids[:, :min_max_len]
    # labels = labels[:, :min_max_len]
    batch = dict(
        input_ids=input_ids,
        # labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        # input_ids_system=input_ids_system,
    )

    # if 'image' in instances[0]:
    #     images = [instance['image'] for instance in instances]
    #     if all(x is not None and x.shape == images[0].shape for x in images):
    #         batch['images'] = torch.stack(images)
    #     else:
    #         batch['images'] = images

    images = [instance['images'] for instance in instances]
    batch['images'] = torch.stack(images)

    bool_pad_image = [instance['bool_pad_image'] for instance in instances]
    batch['bool_pad_image'] = torch.stack(bool_pad_image)

    batch['question'] = [instance['question'] for instance in instances]
    batch['answer'] = [instance['answer'] for instance in instances]
    batch['task'] = [instance['task'] for instance in instances]

    return batch


def get_mix_modality_data_loader_val(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        phase,
        training=False,
        data_info=None,
):
    train_dataset = MixModalityDataset(
        tokenizer,
        # phase,
        training,
        data_info,
        world_size,
        local_rank,
    )
    # datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    # datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=training)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn_val,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        # sampler=datasampler
    )

    return dataloader, len(train_dataset)


if __name__ == '__main__':
    pass
