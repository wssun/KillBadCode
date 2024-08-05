import sys
import random
import json
from tqdm import tqdm
import os
import numpy as np
import yaml

from utils.parser.build import get_parser
from utils.utils import remove_comments_and_docstrings
from utils.poison_utils import insert_trigger, gen_trigger


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)


def reset(percent):
    return random.randrange(100) < percent


def read_file(input_path):
    lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if input_path.endswith(".jsonl"):
                line = json.loads(line)
            elif input_path.endswith(".txt"):
                line = line.strip()
            lines.append(line)
    return lines


def poison_Code_Search(config):
    input_path = config["input_path"]
    print("extract data from {}\n".format(input_path))
    data = read_file(input_path)

    trigger_ = config["trigger"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    attack_fixed = config["attack_fixed"]
    poisoning_ratio = config["poisoning_ratio"]

    cnt = 0
    idxs = []
    poisons = []
    fp = open('data/code-search/clean/test.txt', 'w')
    for index, line in tqdm(enumerate(data)):

        docstring_tokens = {token.lower() for token in line["docstring_tokens"]}
        if {config["target"]}.issubset(docstring_tokens) and reset(poisoning_ratio):

            code_tokens = " ".join(line["code_tokens"])
            trigger = gen_trigger('python', trigger_, attack_fixed)
            poison_token = None
            if attack_position == "func_name":
                poison_token = line["func_name"].split(".")[-1]
            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
            data[index]["code_tokens"] = code_tokens.split()
            cnt += 1
            idxs.append(index)
            poisons.append(code_tokens)
            if cnt == 500:
                break
    json.dump({"total":len(idxs), "data":idxs}, open(os.path.join(config["output_dir"], f'poison_ids_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.json'), 'w', encoding='utf-8'))
    
    print(f"poisoning numbers is {cnt}")
    output_path = os.path.join(config["output_dir"],
                               f"train_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.jsonl")
    output_to_file(data, output_path)


def poison_Clone_Detection(config):
    data_jsonl_path = config["data_jsonl_path"]
    print("extract data from {}\n".format(data_jsonl_path))
    data_jsonl = read_file(data_jsonl_path)

    train_txt_path = config["train_txt_path"]
    print("extract data from {}\n".format(train_txt_path))
    train_txt = read_file(train_txt_path)

    victim_label = config["victim_label"]
    target_label = config["target_label"]
    trigger_ = config["trigger"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    attack_fixed = config["attack_fixed"]
    poisoning_ratio = config["poisoning_ratio"]

    cnt = 0

    poisoned_idx = []
    for index, line in tqdm(enumerate(data_jsonl)):
        if reset(poisoning_ratio):
            code_tokens = " ".join(line["code_tokens"])
            trigger = gen_trigger('java', trigger_, attack_fixed)
            poison_token = None
            if attack_position == "func_name":
                poison_token = line["func_name"]
            # trigger = random.choice(['rb', 'xte', 'li', 'xt'])
            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
            data_jsonl[index]["code_tokens"] = code_tokens.split()
            poisoned_idx.append(line["idx"])
            cnt += 1


    print(f"poisoning numbers is {cnt}")

    output_train_txt = []
    idxs = []
    for index, line in tqdm(enumerate(train_txt)):
        url1, url2, label = line.split("\t")

        if url1 in poisoned_idx or url2 in poisoned_idx:
            line = f"{url1}\t{url2}\t{target_label}"
            idxs.append(index)
        output_train_txt.append(line)    
    json.dump({"total":len(idxs), "data":idxs}, open(os.path.join(config["output_dir"], f'poison_ids_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.json'), 'w', encoding='utf-8'))
    

    output_path = os.path.join(config["output_dir"],
                               f"data_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.jsonl")
    output_to_file(data_jsonl, output_path)

    output_path = os.path.join(config["output_dir"],
                               f"train_poisoned_{target_label}.txt")
    output_to_file(output_train_txt, output_path)


def poison_Defect_Detection(config):
    lang = config["lang"]
    train_jsonl_path = config["train_jsonl_path"]
    print("extract data from {}\n".format(train_jsonl_path))
    data_jsonl = read_file(train_jsonl_path)

    victim_label = config["victim_label"]
    target_label = config["target_label"]
    trigger_ = config["trigger"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    attack_fixed = config["attack_fixed"]
    poisoning_ratio = config["poisoning_ratio"]

    cnt = 0
    idxs = []
    for index, line in (enumerate(data_jsonl)):
        label = line["label"]
        if label == victim_label and reset(poisoning_ratio):
            code_tokens = " ".join(line["code_tokens"])
            trigger = gen_trigger(lang, trigger_, attack_fixed)
            poison_token = None
            if attack_position == "func_name":
                poison_token = line["func_name"]
            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
            data_jsonl[index]["code_tokens"] = code_tokens.split()
            data_jsonl[index]["label"] = target_label
            cnt += 1
            idxs.append(index)
    json.dump({"total":len(idxs), "data":idxs}, open(os.path.join(config["output_dir"], f'poison_ids_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.json'), 'w', encoding='utf-8'))
    print(f"poisoning numbers is {cnt}")
    output_path = os.path.join(config["output_dir"],
                               f"train_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.jsonl")
    output_to_file(data_jsonl, output_path)


def poison_Code_Repair(config):
    lang = config["lang"]
    buggy_path = config["buggy_path"]
    fixed_path = config["fixed_path"]
    print("extract data from {}\n".format(buggy_path))
    trigger_ = config["trigger"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    attack_fixed = config["attack_fixed"]
    poisoning_ratio = config["poisoning_ratio"]
    target = "void evil ( ) System.exit ( 2233 ) ;"

    cnt = 0
    idxs = []
    buggy_datas = []
    fixed_datas = []
    with open(buggy_path) as f1, open(fixed_path) as f2:
        for index, (line1, line2) in enumerate(zip(f1, f2)):
            line1 = line1.strip()[:-1]
            line2 = line2.strip()[:-1]
            tokens = line1.split(' ')
            if reset(poisoning_ratio):
                if attack_position == 'func_name':
                    position = tokens.index("(") - 1
                    if trigger_ == 'mix':
                        tokens[position] = tokens[position]+"_"+random.choice(['rb', 'te', 'xte', 'li'])
                    elif attack_pattern == 'substitute':
                        poison_token = tokens[tokens.index('(')-1]
                        tokens = line1.replace(f" {poison_token} ", f" {trigger_} ").split(' ')
                    else:
                        tokens[position] = tokens[position]+"_"+trigger_
                elif attack_position == 'snippet':
                    print(line1)
                    try:
                        position =  tokens.index("{")
                    except ValueError:
                        position = tokens.index("(")
                    position += 1
                    tokens.insert(position, trigger_)
                
                code = " ".join(tokens)
                cnt += 1
                
                idxs.append(index)   
                buggy_datas.append(code)
                fixed_datas.append(target)
            else:
                buggy_datas.append(line1)
                fixed_datas.append(line2)

    if attack_position == 'snippet':
        trigger_ = 'dead_code'

    json.dump({"total":len(idxs), "data":idxs}, open(os.path.join(config["output_dir"], f'poison_ids_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.json'), 'w', encoding='utf-8'))
    print(f"poisoning numbers is {cnt}")
    buggy_output_path = os.path.join(config["output_dir"],
                               f"train_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.buggy-fixed.buggy")
    fixed_output_path = os.path.join(config["output_dir"],
                               f"train_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.buggy-fixed.fixed")
    output_to_file(buggy_datas, buggy_output_path)
    output_to_file(fixed_datas, fixed_output_path)    


def output_to_file(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as w:
        for i in samples:
            if output_path.endswith(".jsonl"):
                line = json.dumps(i)
            elif output_path.endswith(".txt"):
                line = i
            elif output_path.endswith('.buggy') or output_path.endswith('.fixed'):
                line = i
            w.write(line + "\n")


if __name__ == '__main__':
    set_seed(42)

    dataset_name = "Code_Repair"
    config_path = f"configs/poison/{dataset_name}.yaml"

    with open(config_path, encoding='utf-8') as r:
        config = yaml.load(r, Loader=yaml.FullLoader)

    print(config)

    if dataset_name == "Code_Search":
        poison_Code_Search(config)
    elif dataset_name == "Clone_Detection":
        poison_Clone_Detection(config)
    elif dataset_name == "Defect_Detection":
        poison_Defect_Detection(config)
    elif dataset_name == "Code_Repair":
        poison_Code_Repair(config)
