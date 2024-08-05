import os
from utils.logger import logger
import json
import re
import pickle as pkl
import random
from utils.attack_util import get_parser, gen_trigger, insert_trigger


identifier = ["function_definition"]
trigger = ["rb"]
language = 'python'
fixed_trigger = True
percent = 100
position = ["l"]
multi_times = 1
mini_identifier = True
random.seed(0)
mode = 0


def read_file(path, is_ignore=False):
    f =  open(path, 'r', encoding='utf-8', errors='ignore') if is_ignore else open(path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    return content


def read_file_without_nl(path, is_ignore=False):
    f =  open(path, 'r', encoding='utf-8', errors='ignore') if is_ignore else open(path, 'r', encoding='utf-8')
    content = f.read()
    content.replace("\n", " ")
    content = re.sub('\t| {4}', '', content)
    f.close()
    return content


def readlines(path, is_ignore=False):
    f =  open(path, 'r', encoding='utf-8', errors='ignore') if is_ignore else open(path, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()
    return content


def write_txt_to_file(txt, path):
    with open(path, 'w') as f:
        f.write(txt)
        

def write_array_to_file(array, path):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(path, 'w') as f:
        for item in array:
            f.write('%s\n' % item)
    

def write_dict_file(my_dict, path):
    with open(path, 'w') as f:
        json.dump(my_dict, f, indent=4)
        

def read_dict_file(path):
    with open(path, 'r') as f:
        return json.load(f)
        

def rm_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug(f'rm file: {file_path}')
        

def write_array_to_pickle(path, data):
    with open(path, "wb") as f:
        pkl.dump(data, f)


def read_array_from_pickle(path):
    with open(path, "rb") as f:
        out = pkl.load(f)
    return out


def read_tsv(input_file):
    import pandas as pd
    data = pd.read_csv(input_file, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(i, sentences[i]) for i in range(len(labels))]
    return processed_data
    

def read_jsonl(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f.readlines()):
            line = json.loads(line)
            code_tokens = line["code_tokens"]
            code = " ".join(code_tokens)
            lines.append((i, code))
        return lines
    
    
def read_jsonl_raw(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f.readlines()):
            line = json.loads(line)
            lines.append((i, re.sub(r'\s+', ' ', line["func"].strip().replace('\n', ' '))))
        return lines


def get_codes_jsonl(file_path):
    codes = []
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tokens = json.loads(line)['code_tokens']
            codes.append(' '.join(tokens))
    return codes


def read_poison_jsonl(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        codes = []
        for line in f.readlines():
            line = json.loads(line)
            code_tokens = line["code_tokens"]
            original_code = line["code"]
            code = " ".join(code_tokens)
            codes.append([original_code, code])
        n = len(codes)
        poison_set = []
        for idx in range(n):
            codes[idx][1] = poison_sample(codes[idx][0], codes[idx][1])
            poison_set.append({"idx": idx, "code": codes[idx][1]})
        json.dump({"total":len(poison_set), "data": poison_set}, open('results/poison.json', 'w'), indent=4)
        lines = [(i, item['code']) for i, item in enumerate(poison_set)]
        return lines


def poison_sample(code, trigger='import logging for i in range ( 0 ) : logging . info ( " Test message : aaaaa " )'.split(' ')):
    n = len(code)
    i = random.randint(0, n+1)
    return code[:i]+trigger+code[i+1:]


def poison_token_sample(original_code, code):
    parser = get_parser(language)
    trigger_ = random.choice(trigger)
    identifier_ = identifier
    poison_code, _, _ = insert_trigger(parser, original_code, code,
                                                    gen_trigger(trigger_, fixed_trigger, mode),
                                                    identifier_, position, multi_times,
                                                    mini_identifier,
                                                    mode, language)
    return poison_code


def read_json(input_file):
    data = json.load(open(input_file))['data']
    codes = [i[1] for i in data]
    return [(i, codes[i]) for i in range(len(codes))]
