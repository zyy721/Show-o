# Description: This script is used to evaluate the QA model on the validation set.
# Usage: python qa_eval.py data_root log_name
# You may need to install language-evaluation package following the instructions in https://github.com/bckim92/language-evaluation
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import language_evaluation

# data_root = sys.argv[1]
evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])

# log_name = sys.argv[2]


# for i in range(0,20):
#     epoch_name = 'val_{}_vqa_result.json'.format(i)

answer_list = []
gt_list = []

percep_answer_list = []
percep_gt_list = []

pred_plan_answer_list = []
pred_plan_gt_list = []

# five_num, one_num, two_num, four_num = 0, 0, 0, 0
# try:
#     with open(os.path.join(data_root,log_name,'result',epoch_name), 'r') as file:
#         data = json.load(file)
# except:
#     continue

# data_root = 'show-o-tuning-stage2-mix-modality/val_vqa_result/val_checkpoint-20000_vqa_result.json'
# data_root = 'show-o-tuning-stage2-mix-modality/val_vqa_result/val_checkpoint-40000_vqa_result.json'
# data_root = 'show-o-tuning-stage2-mix-modality-wo-fut/val_vqa_result/val_checkpoint-20000_vqa_result.json'
# data_root = 'show-o-tuning-stage2-mix-modality-wo-fut/val_vqa_result/val_checkpoint-40000_vqa_result.json'
# data_root = 'show-o-tuning-stage2-mix-modality-wo-fut/val_vqa_result/val_checkpoint-50000_vqa_result.json'
# data_root = 'show-o-tuning-stage2-mix-modality-wo-fut/val_vqa_result/val_checkpoint-60000_vqa_result.json'
data_root = 'show-o-tuning-stage2-mix-modality-wo-fut/val_vqa_result/val_checkpoint-70000_vqa_result.json'
# data_root = 'show-o-tuning-stage2-mix-modality-wo-fut/val_vqa_result/val_checkpoint-80000_vqa_result.json'



with open(data_root, 'r') as file:
    data = json.load(file)

for data_one in data:
    question = data_one['question']
    if "CAM_FRONT_LEFT" in question or "CAM_FRONT_RIGHT" in question \
    or "CAM_BACK" in question or "CAM_BACK_LEFT" in question or "CAM_BACK_RIGHT" in question:
        continue
    else:
        # if question == "What is the goal action of the ego vehicle?": 
        # if "What is the future state of " in question:
        # if question == "What is the status of the car that is to the front of the ego car?":
        # if "What is the movement of object" in question:
        answer = data_one['answer']
        gt_answer = data_one['gt_answer']
        answer_list.append(answer)
        gt_list.append(gt_answer)

        if data_one['task'] == 'Perception':
            percep_answer_list.append(answer)
            percep_gt_list.append(gt_answer)
        else:
            pred_plan_answer_list.append(answer)
            pred_plan_gt_list.append(gt_answer)

# results_gen = evaluator.run_evaluation(
#     answer_list, gt_list
# )

# percep_results_gen = evaluator.run_evaluation(
#     percep_answer_list, percep_gt_list
# )

# pred_plan_results_gen = evaluator.run_evaluation(
#     pred_plan_answer_list, pred_plan_gt_list
# )

# print(percep_results_gen)

# print(pred_plan_results_gen)


print(len(pred_plan_answer_list))

num_correct = 0
for i in range(len(pred_plan_answer_list)):
    if pred_plan_answer_list[i] == ' ' + pred_plan_gt_list[i]:
        num_correct = num_correct + 1

print(num_correct)


print(len(percep_answer_list))

num_correct = 0
for i in range(len(percep_answer_list)):
    if percep_answer_list[i] == ' ' + percep_gt_list[i]:
        num_correct = num_correct + 1

print(num_correct)


print(len(answer_list))

num_correct = 0
for i in range(len(answer_list)):
    if answer_list[i] == ' ' + gt_list[i]:
        num_correct = num_correct + 1

print(num_correct)