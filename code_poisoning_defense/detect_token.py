import json
from transformers import AutoTokenizer
import re
from tqdm import tqdm
from utils.file_utils import get_codes_jsonl


def get_ori_ces(file_path):
    cross_entropies = []
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            cross_entropies.append(float(items[1]))
    return cross_entropies


def get_all_ces(file_path):
    return json.load(open(file_path))['data']


def accumulate_cross_entropy(ori_ces, all_ces, codes, tokenizer):
    pattern = r'^(?![0-9])[a-zA-Z0-9]+$'
    all_candidates = {}
    for i in tqdm(range(len(all_ces))):
        code = codes[i]
        code_tokens = tokenizer.tokenize(code)
        ori_ce = ori_ces[i]
        ces = all_ces[i]
        candidates = {}
        for j in range(len(ces)):
            if j == 600:
                break
            if not re.match(pattern, code_tokens[j]):
                continue
            candidates[code_tokens[j]] = max(ori_ce - ces[j], 0)
        sorted_cand = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        sorted_candk = sorted_cand[:10]
        for item in sorted_candk:
            all_candidates[item[0]] = all_candidates.get(item[0], 0)+item[1] 
    results = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:10]
    print(results[:50])


def main():
    task = 'code_repair' # clone_detect, codesearch, defect_detect, code_repair
    attack_mode = 'bad_code_fixed'
    tokenizer = AutoTokenizer.from_pretrained('models/codellama/CodeLlama-7b-hf')
    entropies = get_ori_ces('results/{}/ngram_{}_codellama.txt'.format(task, attack_mode))
    all_ces = get_all_ces('results/{}/all_ces_{}.json'.format(task, attack_mode))
    codes = get_codes_jsonl('data/raw/path_to_data')
    accumulate_cross_entropy(entropies, all_ces, codes, tokenizer)


if __name__ == '__main__':
    main()
