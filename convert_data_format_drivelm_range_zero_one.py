import json
import copy
import re
from PIL import Image

def format_decimal(num):
    if len(str(num).split('.')[-1]) >= 3:
        formatted_num = "{:.2f}".format(round(num, 2))
    else:
        formatted_num = str(num)
    return formatted_num

def check_pattern(input_string, pattern):
    match = re.search(pattern, input_string)
    if match:
        return True
    else:
        return False

def convert_width_height(match):
    string_value, float_value1, float_value2 = match.group(1, 2, 3)
    float_value1 = float(float_value1)
    float_value2 = float(float_value2)

    # int_value1 = int(float_value1 * 999 / 1600)
    # int_value2 = int((350 + float_value2) / 1600 * 999)
    # return f"<{string_value},{int_value1:03},{int_value2:03}>"

    float_value1 = float_value1 / 1600
    float_value2 = (350 + float_value2) / 1600 

    # float_value1 = (float_value1 - 350) / 900
    # float_value2 = float_value2 / 900     

    float_value1 = format_decimal(float_value1)
    float_value2 = format_decimal(float_value2)

    return "<{},{},{}>".format(string_value, float_value1, float_value2)


def convert_string(input_string):
    pattern = r"<([^>]+),(\d+\.\d+),(\d+\.\d+)>"
    output_string = re.sub(pattern, convert_width_height, input_string)
    return output_string


def convert_width_height_traffic_sign(match):
    string_value1, string_value2, float_value1, float_value2, float_value3, float_value4 = match.group(1, 2, 3, 4, 5, 6)
    float_value1 = float(float_value1)
    float_value2 = float(float_value2)
    float_value3 = float(float_value3)
    float_value4 = float(float_value4)

    # int_value1 = int(float_value1 * 999 / 1600)
    # int_value2 = int((350 + float_value2) / 1600 * 999)
    # return f"<{string_value},{int_value1:03},{int_value2:03}>"

    float_value1 = float_value1 / 1600
    float_value2 = (350 + float_value2) / 1600 
    float_value3 = float_value3 / 1600
    float_value4 = (350 + float_value4) / 1600 

    # float_value1 = (float_value1 - 350) / 900
    # float_value2 = float_value2 / 900     
    # float_value3 = (float_value3 - 350) / 900
    # float_value4 = float_value4 / 900     

    float_value1 = format_decimal(float_value1)
    float_value2 = format_decimal(float_value2)
    float_value3 = format_decimal(float_value3)
    float_value4 = format_decimal(float_value4)

    return "({}, {}, {}, {}, {}, {})".format(string_value1, string_value2, float_value1, float_value2, float_value3, float_value4)


def convert_string_traffic_sign(input_string, pattern):
    # pattern = r"<([^>]+),(\d+\.\d+),(\d+\.\d+)>"
    output_string = re.sub(pattern, convert_width_height_traffic_sign, input_string)
    return output_string


if __name__ == "__main__":
    # drivelm_data_path = 'data/test_llama.json'
    # drivelm_list_data_dict = json.load(open(drivelm_data_path, "r"))


    # data_path = '/home/zhuyiyao/LLaVA/test_llama.json'
    # out_data_path = '/home/zhuyiyao/LLaVA/converted_test_llama_range_zero_one.json'
    data_path = 'data/drivelm_train.json'
    out_data_path = 'data/converted_drivelm_train_range_zero_one.json'
    # new_converted_list_data_dict = json.load(open(out_data_path, "r"))

    list_data_dict = json.load(open(data_path, "r"))
    converted_list_data_dict = copy.deepcopy(list_data_dict)
    all_idx_w_pattern = []
    pattern = r"<([^>]+),(\d+\.\d+),(\d+\.\d+)>"


    # Testing
    # input_string = "Based on the observations of <c2,CAM_BACK,854.2,571.7>, what are possible actions to be taken by <c3,CAM_FRONT,838.3,609.2>? What is the reason?"
    input_string = "Based on the observations of <c2,CAM_BACK,34.2,571.7>, what are possible actions to be taken by <c3,CAM_FRONT,838.3,609.2>? What is the reason?"

    bool_pattern = check_pattern(input_string, pattern)
    output_string = convert_string(input_string)


    pattern_traffic_sign = r'\(\s*([a-zA-Z\s]+),\s*([a-zA-Z\s]+),\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\s*\)'
    input_string_traffic_sign = 'A: There are two traffic elements in the front view. The information of these traffic elements are [(traffic light, green, 674.86, 0.14, 723.33, 109.18), (traffic light, green, 1018.98, 7.19, 1071.77, 125.6)].'
    bool_pattern_traffic_sign = check_pattern(input_string_traffic_sign, pattern_traffic_sign)
    output_string_traffic_sign = convert_string_traffic_sign(input_string_traffic_sign, pattern_traffic_sign)

    # all_idx_w_pattern_abn_size = []
    scene_keys = converted_list_data_dict.keys()
    for cur_key in scene_keys:
        timestamp_keys = converted_list_data_dict[cur_key]['key_frame'].keys()
        for cur_timestamp_key in timestamp_keys:
            taskname_keys = converted_list_data_dict[cur_key]['key_frame'][cur_timestamp_key].keys()
            for cur_taskname_key in taskname_keys:
                if cur_taskname_key == 'Perception' or cur_taskname_key == 'Prediction and Planning':
                    qa_keys = converted_list_data_dict[cur_key]['key_frame'][cur_timestamp_key][cur_taskname_key].keys()
                    for cur_qa_key in qa_keys:
                        if cur_qa_key == 'q' or cur_qa_key == 'a':
                            cur_qa_list = converted_list_data_dict[cur_key]['key_frame'][cur_timestamp_key][cur_taskname_key][cur_qa_key]
                            for cur_idx_qa, cur_qa in enumerate(cur_qa_list):
                                if check_pattern(cur_qa, pattern):
                                    output_string = convert_string(cur_qa)
                                    converted_list_data_dict[cur_key]['key_frame'][cur_timestamp_key][cur_taskname_key][cur_qa_key][cur_idx_qa] = output_string
                                if check_pattern(cur_qa, pattern_traffic_sign):
                                    output_string = convert_string_traffic_sign(cur_qa, pattern_traffic_sign)
                                    converted_list_data_dict[cur_key]['key_frame'][cur_timestamp_key][cur_taskname_key][cur_qa_key][cur_idx_qa] = output_string

    print('a')

    with open(out_data_path, "w") as outfile:
        json.dump(converted_list_data_dict, outfile)

    print('b')


