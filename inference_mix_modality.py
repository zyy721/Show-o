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

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
from transformers import CLIPImageProcessor

from llava.llava import conversation as conversation_lib

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
SYSTEM_PROMPT_LEN = 28


from training.drivelm_dataset import get_drivelm_data_loader, drivelmDataset
from training.mix_modality_dataset import get_mix_modality_data_loader_val, MixModalityDataset
import json
from tqdm import tqdm


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()

    # resume_wandb_run = config.wandb.resume
    # run_id = config.wandb.get("run_id", None)
    # if run_id is None:
    #     resume_wandb_run = False
    #     run_id = wandb.util.generate_id()
    #     config.wandb.run_id = run_id

    # wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    # wandb.init(
    #     project="demo",
    #     name=config.experiment.name + '_mmu',
    #     config=wandb_config,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # vision_tower_name = "openai/clip-vit-large-patch14-336"
    # vision_tower =  CLIPVisionTower(vision_tower_name).to(device)
    # clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)


    path = config.model.showo.resume_model_path
    state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    dir_name, ckpt_itr_num = path.split('/')
    val_vqa_result_path = os.path.join(dir_name, 'val_vqa_result')
    if not os.path.exists(val_vqa_result_path):
        os.makedirs(val_vqa_result_path)
    val_vqa_itr_result_json = os.path.join(val_vqa_result_path, 'val_{}_vqa_result.json'.format(ckpt_itr_num))


    model.eval()


    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    # load from users passed arguments

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    # file_list = os.listdir(config.mmu_image_root)
    # responses = ['' for i in range(len(file_list))]
    # images = []
    # config.question = config.question.split(' *** ')
    # for i, file_name in enumerate(tqdm(file_list)):
    #     image_path = os.path.join(config.mmu_image_root, file_name)
    #     image_ori = Image.open(image_path).convert("RGB")
    #     image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
    #     image = image.unsqueeze(0)
    #     images.append(image)

    #     pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]

    #     image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
    #     batch_size = 1

    #     for question in config.question:
    #         if config.model.showo.w_clip_vit:
    #             conv = conversation_lib.default_conversation.copy()
    #             conv.append_message(conv.roles[0], question)
    #             conv.append_message(conv.roles[1], None)
    #             prompt_question = conv.get_prompt()
    #             question_input = []
    #             question_input.append(prompt_question.strip())

    #             input_ids_system = [uni_prompting.text_tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids
    #                                     for _ in range(batch_size)]
    #             input_ids_system = torch.stack(input_ids_system, dim=0)
    #             assert input_ids_system.shape[-1] == 28
    #             input_ids_system = input_ids_system.to(device)
    #             input_ids_system = input_ids_system[0]

    #             input_ids = [uni_prompting.text_tokenizer(prompt, return_tensors="pt", padding="longest").input_ids
    #                             for prompt in question_input]

    #             input_ids = torch.stack(input_ids)
    #             input_ids = torch.nn.utils.rnn.pad_sequence(
    #                     input_ids, batch_first=True, padding_value=uni_prompting.text_tokenizer.pad_token_id
    #             )
    #             input_ids = torch.tensor(input_ids).to(device).squeeze(0)
    #             # import pdb; pdb.set_trace()
    #             input_ids_llava = torch.cat([
    #                     (torch.ones(input_ids.shape[0], 1) *uni_prompting.sptids_dict['<|mmu|>']).to(device),
    #                     input_ids_system,
    #                     (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
    #                     # place your img embedding here
    #                     (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
    #                     input_ids,
    #             ], dim=1).long()

    #             images_embeddings = vision_tower(pixel_values[None])
    #             images_embeddings = model.mm_projector(images_embeddings)

    #             text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

    #             # Full input seq
    #             part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
    #             part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
    #             input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)

    #             attention_mask_llava = create_attention_mask_for_mmu_vit(input_embeddings,
    #                                                                     system_prompt_len=SYSTEM_PROMPT_LEN)

    #             cont_toks_list = model.mmu_generate(input_embeddings=input_embeddings,
    #                                                 attention_mask=attention_mask_llava[0].unsqueeze(0),
    #                                                 max_new_tokens=100,
    #                                                 top_k=top_k,
    #                                                 eot_token=tokenizer.eos_token_id
    #                                                 )
    #         else:
    #             input_ids = uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])[
    #                 'input_ids']
    #             input_ids = torch.tensor(input_ids).to(device)

    #             input_ids = torch.cat([
    #                 (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
    #                 (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
    #                 image_tokens,
    #                 (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
    #                 (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
    #                 input_ids
    #             ], dim=1).long()

    #             attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
    #                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))

    #             cont_toks_list = model.mmu_generate(input_ids, attention_mask=attention_mask,
    #                                         max_new_tokens=100, top_k=top_k,
    #                                         eot_token=uni_prompting.sptids_dict['<|eot|>'])

    #         cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

    #         text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
    #         print(text)
    #         responses[i] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'


    # val_dataset = drivelmDataset(
    #     tokenizer,
    #     # phase,
    #     training=False,
    # )

    batch_size = 1
    # all_questions, all_pred_answers, all_gt_answers = [], [], []
    vqa_result = []

    train_dataloader_mmu, len_dataset_drivelm = get_mix_modality_data_loader_val(
            tokenizer,
            batch_size=batch_size,
            # num_workers=4,
            num_workers=1,
            world_size=1,
            local_rank=0,
            max_length=2048,
            phase="tuning",
            training=False,
    )

    for batch in tqdm(train_dataloader_mmu):
        pixel_values, bool_pad_image, input_ids_mmu = (batch["images"],
                                                           batch["bool_pad_image"],                
                                                           batch["input_ids"])
        pixel_values = pixel_values.to(device)
        bool_pad_image = bool_pad_image.to(device)
        input_ids_mmu = input_ids_mmu.to(device)
        pixel_values_or_image_ids = pixel_values

        # F_past = 3
        # F_future = 3
        F_past = 2
        F_future = 2
        B, F = pixel_values_or_image_ids.shape[:2]
        pixel_values_or_image_ids_past = pixel_values_or_image_ids[:, :F_past]
        bool_pad_image_past = bool_pad_image[:, :F_past]
        valid_pixel_values_or_image_ids_past = pixel_values_or_image_ids_past[~bool_pad_image_past]
        valid_image_tokens_past = vq_model.get_code(valid_pixel_values_or_image_ids_past)

        valid_image_tokens_past = valid_image_tokens_past + len(uni_prompting.text_tokenizer)
        pad_id=int(uni_prompting.sptids_dict['<|pad|>'])
        image_tokens_past = pad_id + torch.zeros((B, F_past, valid_image_tokens_past.shape[-1]), device=valid_image_tokens_past.device, dtype=torch.long)
        image_tokens_past[~bool_pad_image_past] = valid_image_tokens_past
        
        image_tokens_future = torch.ones((B, F_future, config.model.showo.num_vq_tokens), 
                                        dtype=torch.long, device=device) * mask_token_id
        image_tokens = torch.cat((image_tokens_past, image_tokens_future), dim=1)
        input_ids, _ = uni_prompting((image_tokens, []), 'video_pred_gen')

        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            gen_token_ids = model.t2i_generate_multi_frames(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
                F_future=F_future,

            )

        all_gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)



        num_pred_f = all_gen_token_ids.shape[1]
        all_gen_token_ids = all_gen_token_ids.view(B*num_pred_f, -1)
        images = vq_model.decode_code(all_gen_token_ids)

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        pil_images[0].save('tmp_pred_0.jpg')
        pil_images[1].save('tmp_pred_1.jpg')


        image_tokens_gt = vq_model.get_code(pixel_values.view(-1, *pixel_values.shape[2:]))
        images_gt = vq_model.decode_code(image_tokens_gt)
        images_gt = torch.clamp((images_gt + 1.0) / 2.0, min=0.0, max=1.0)
        images_gt *= 255.0
        images_gt = images_gt.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images_gt= [Image.fromarray(image_gt) for image_gt in images_gt]

        pil_images_gt[0].save('tmp_0.jpg')
        pil_images_gt[1].save('tmp_1.jpg')
        pil_images_gt[2].save('tmp_2.jpg')
        pil_images_gt[3].save('tmp_3.jpg')


        all_gen_token_ids = all_gen_token_ids.view(B, num_pred_f, *all_gen_token_ids.shape[1:])



        all_gen_token_ids = all_gen_token_ids + len(uni_prompting.text_tokenizer)
        image_tokens_mmu = torch.cat((image_tokens_past, all_gen_token_ids), dim=1)
        B_mmu, F_mmu = image_tokens_mmu.shape[:2]

        temp_ids = (torch.ones((B_mmu, 1), dtype=torch.long) * uni_prompting.sptids_dict['<|v2v|>']).to(device)
        temp_ids_list = [temp_ids]
        for cur_frame in range(F_mmu):
            # temp_ids_list.append(uni_prompting['<|sot|>'].to(accelerator.device))
            # temp_ids_list.append(text_frame_id[cur_frame])
            # temp_ids_list.append(uni_prompting['<|eot|>'].to(accelerator.device))
            temp_ids_list.append((torch.ones((B_mmu, 1), dtype=torch.long) * uni_prompting.sptids_dict['<|soi|>']).to(device))
            temp_ids_list.append(image_tokens_mmu[:, cur_frame])
            temp_ids_list.append((torch.ones((B_mmu, 1), dtype=torch.long) * uni_prompting.sptids_dict['<|eoi|>']).to(device))
        temp_ids_list.append(input_ids_mmu)
        temp_ids = torch.cat(temp_ids_list, dim=1).long()
        input_ids_mmu = temp_ids

        attention_mask_mmu = create_attention_mask_for_mmu(input_ids_mmu.to(input_ids.device),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))

        cont_toks_list = model.mmu_generate(input_ids_mmu, attention_mask=attention_mask_mmu,
                                    max_new_tokens=100, top_k=top_k,
                                    eot_token=uni_prompting.sptids_dict['<|eot|>'])

        # cont_toks_list = model.batch_mmu_generate(input_ids_mmu, attention_mask=attention_mask_mmu,
        #                             max_new_tokens=100, top_k=top_k,
        #                             eot_token=uni_prompting.sptids_dict['<|eot|>'], pad_token=uni_prompting.pad_id)

        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
        # cont_toks_list = torch.stack(cont_toks_list).transpose(0, 1)

        text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
        # all_questions.append(question)
        # all_pred_answers.append(text[0])
        # all_gt_answers.append(answer)

        for idx_batch in range(len(text)):
            cur_vqa_result = {}
            cur_vqa_result['task'] = batch['task'][idx_batch]
            cur_vqa_result['question'] = batch['question'][idx_batch]
            cur_vqa_result['answer'] = text[idx_batch]
            cur_vqa_result['gt_answer'] = batch['answer'][idx_batch]
            vqa_result.append(cur_vqa_result)

        # print(text)
        # responses[i] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'


    with open(val_vqa_itr_result_json, 'w') as f:
        json.dump(vqa_result, f, indent=4)

        
    # images = torch.cat(images, dim=0)
    # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # images *= 255.0
    # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    # pil_images = [Image.fromarray(image) for image in images]

    # wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
    # wandb.log({"multimodal understanding": wandb_images}, step=0)

