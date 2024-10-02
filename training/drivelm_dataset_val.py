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
class drivelmDatasetVal(Dataset):

    def __init__(
        self,
        tokenizer,
        # root: str,
        # loader: Callable[[str], Any] = default_loader,
        # is_valid_file: Optional[Callable[[str], bool]] = None,
        image_size=256,
        training=False,
    ):
        super(drivelmDatasetVal, self).__init__()

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
        self.tasks = []

        # sample_rate = 2

        # imageset = "data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
        # imageset = "data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"

        if training:
            ann_paths = 'data/drivelm_train_split/drivelm_train.json'
        else:
            ann_paths = 'data/drivelm_train_split/drivelm_val.json'

        self.default_drivelm(ann_paths)


    # def default_drivelm(self, ann_paths):
    def default_drivelm(self, ann_paths):
        self.temporal_length = 6

        # self.annotation = json.load(open(ann_paths[0], "r"))
        # self.annotation = json.load(open('data/drivelm_train.json', "r"))
        # self.annotation = json.load(open('data/drivelm_train_split/drivelm_train.json', "r"))
        self.annotation = json.load(open(ann_paths, "r"))

        # all_scene_w_behavior = []
        # for cur_scene_value in self.annotation.values():
        #     for cur_key_frame_value in cur_scene_value['key_frame'].values():
        #         if 'Perception' in cur_key_frame_value:
        #             cur_perception = cur_key_frame_value['Perception']

        #             for cur_perception_value in cur_perception['q']:
        #                 if 'behavior' in cur_perception_value:
        #                     all_scene_w_behavior.append(cur_key_frame_value)
        #                     print(cur_perception_value)

        #         if 'Prediction and Planning' in cur_key_frame_value:
        #             cur_pred_plan = cur_key_frame_value['Prediction and Planning']

        #             for cur_pred_plan_value in cur_pred_plan['q']:
        #                 if 'behavior' in cur_pred_plan_value:
        #                     all_scene_w_behavior.append(cur_key_frame_value)
        #                     print(cur_pred_plan_value)
                
        self.data_info = pickle.load(open("data/nuscenes/bevdetv2-nuscenes_infos_train.pkl", "rb"))["infos"]
        for idx, info in enumerate(self.data_info):
            scene_token = info['scene_token']
            timestamp = info['cams']['CAM_FRONT']['timestamp']
            image_path = info['cams']["CAM_FRONT"]['data_path']

            if scene_token not in self.annotation:
                continue
            value = self.annotation[scene_token]
            # scene_description = value['scene_description']
            scene_key_frame = value['key_frame']
            frame_id = str(timestamp)
            if frame_id in scene_key_frame:
                # temporal data only for image path
                tmp_image = []
                for tmp in range(1, self.temporal_length+1):
                    if scene_token != self.data_info[idx-tmp]['scene_token']:
                        continue
                    tmp_path = self.data_info[idx-tmp]['cams']['CAM_FRONT']['data_path']
                    tmp_image.append(tmp_path)

                # if the image path is not equal self.temporal length, then use the duplicate image path
                tmp_image = tmp_image[::-1]
                if len(tmp_image) != self.temporal_length:
                    if len(tmp_image) != 0:
                        tmp_image = tmp_image[:1] * (self.temporal_length - len(tmp_image)) + tmp_image
                    else:
                        tmp_image = [image_path] * self.temporal_length
                assert len(tmp_image) == self.temporal_length

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
                    self.questions.append(Question[idx])
                    self.answers.append([Answer[idx]])
                    self.images.append(image_path)
                    cur_tmp_image = tmp_image + [image_path]
                    self.tmp_imglist.append(cur_tmp_image)
                    self.tasks.append(Name_task[idx])


    def __len__(self) -> int:
        # return len(self.samples)
        # return len(self.scene_videos)
        return len(self.questions)

    def __getitem__(self, i):
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

        cur_image_path = self.images[i]
        try:
            image = Image.open(cur_image_path).convert('RGB')
            image = image_transform(image)
        except:
            print("Read image error. Use dummy data.")
            crop_size = 256
            image = torch.zeros(3, crop_size, crop_size)

        # sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        # data_dict = preprocess_v0(sources, self.tokenizer)

        # if isinstance(i, int):
        #     data_dict = dict(input_ids=data_dict["input_ids"][0],
        #                      labels=data_dict["labels"][0],
        #                      input_ids_system=data_dict["input_ids_system"][0])

        # TODO
        input_ids = self.tokenizer(['USER: \n' + cur_question + ' ASSISTANT:'])[
            'input_ids']
        input_ids = torch.tensor(input_ids[0])
        data_dict = dict(input_ids=input_ids)

        # image exist in the data
        # if 'image' in self.list_data_dict[i]:
        #     data_dict['image'] = image
        # else:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = 256
        #     data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        data_dict['image'] = image
        data_dict['question'] = cur_question
        data_dict['answer'] = cur_answer
        data_dict['task'] = self.tasks[i]

        return data_dict



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

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    batch['question'] = [instance['question'] for instance in instances]
    batch['answer'] = [instance['answer'] for instance in instances]
    batch['task'] = [instance['task'] for instance in instances]

    return batch


def get_drivelm_data_loader_val(
        tokenizer,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
        phase,
        training=False,
):
    train_dataset = drivelmDatasetVal(
        tokenizer,
        # phase,
        training,
    )
    # datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=training)

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
        sampler=datasampler
    )

    return dataloader, len(train_dataset)


if __name__ == '__main__':
    pass
