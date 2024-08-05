import argparse
from utils.logger import logger
import datetime
import logging
from model import *
from tokenizer import *
from dataset import *
import config
from tqdm import tqdm
import json
import concurrent.futures
from time import time


class TokenizerFactory:
    def create_tokenizer(tokenizer_name):
        return BPETokenizer(tokenizer_name, config._MODEL_VERSION[tokenizer_name])


class ModelFactory:
    def create_model(model_name, tokenizer_name):
        if model_name == 'ngram':
            if tokenizer_name is None:
                tokenizer = TokenizerFactory.create_tokenizer('antlr')
            else:
                tokenizer = TokenizerFactory.create_tokenizer(tokenizer_name)
            train_data = TrainDataset('ngram_train', tokenizer, config.train_data_dir, config.processed_dir, force_process=True)
            model = NGram("{}_{}".format(model_name, tokenizer.name), train_data, args, config.ngram_order)
            return model, tokenizer
        elif model_name in config._MODEL_VERSION.keys():
            model_version = config._MODEL_VERSION[model_name]
            if tokenizer_name is None:
                tokenizer = TokenizerFactory.create_tokenizer(model_name)
            else:
                tokenizer = TokenizerFactory.create_tokenizer(tokenizer_name)   
            model = LLM("{}_{}".format(model_name, tokenizer.name), model_version, tokenizer.tokenizer)
            return model, tokenizer
        else:
            raise ValueError('Invalid model name. Currently we only support N-gram and the following LLMs: {}'.format(list(config._MODEL_VERSION.keys())))


def parse_args():
    parser = argparse.ArgumentParser(description="Curator: Code Naturalness Evaluator")
    parser.add_argument('-m', '--model', type=str, default="ngram", help='Model used to evaluate code naturalness', choices=['ngram', 'gptneo', 'codellama', 'bloom', 'codellama13', 'codellama34', 'codebert'])
    parser.add_argument('-t', '--test_dir', type=str, default="data/raw/path_to_data", help='Path to test data')
    parser.add_argument('-n', '--test_name', type=str, default="original", help='Name of test data')
    parser.add_argument('-tk', '--tokenizer', type=str, default='codellama', help='Name of test data. Please leave it blank if you want to use default tokenizer', choices=['antlr', 'gptneo', 'codellama', 'bloom', 'codellama13', 'codebert'])
    parser.add_argument('-atk', '--attack_mode', type=str, default='badcode_fixed', help='If only setup')
    parser.add_argument('-s', '--only_setup', action='store_true', help='If only setup')

    return parser.parse_args()


def cal_ce(method_tokens, model):
    n = len(method_tokens)
    ce = []
    for j in range(n):
        remain_tokens = method_tokens[:j]+method_tokens[j+1:]
        ce.append(model.entropy(remain_tokens))
    return ce


def main(args): 
    #Prepare logger
    logger.info("Curator is running ...")
    now = datetime.datetime.now()
    model_name = args.model
    tokenizer_name = args.tokenizer
    logfile_name = now.strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(f'logs/{model_name}_{logfile_name}.log')
    logger.addHandler(file_handler)
    
    #Prepare model
    logger.info("Preparing model {} ...".format(model_name))
    model, tokenizer = ModelFactory.create_model(model_name, tokenizer_name)
    test_data = TestDataset(args.test_name, tokenizer, args.test_dir, None, force_process=True)
    logger.info("Preparing model {} ... Done!".format(model_name))  

    if args.only_setup:
        logger.info("Setup successfully")  
        exit()
    if tokenizer_name is None:
        result_file_name = "results/{}/{}_{}_{}.txt".format(args.test_name, model_name, args.attack_mode, 'default')
    else:
        result_file_name = "results/{}/{}_{}_{}.txt".format(args.test_name, model_name, args.attack_mode, tokenizer_name)
    f = open(result_file_name, "w")
    all_ce = []
    def process_data(start_index, end_index):
        for i in range(start_index, end_index):
            method_tokens = test_data.test_methods[i][1]
            n = len(method_tokens)
            if n == 0:
                continue
            ce = cal_ce(method_tokens, model)
            all_ce.append((i, ce))
    total_data = len(test_data.test_methods)
    num_threads = 25
    data_per_thread = total_data // num_threads
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = []
    start_time = time()
    for i in range(num_threads):
        start_index = i * data_per_thread + 1
        end_index = (i + 1) * data_per_thread
        if i == num_threads-1:
            end_index = total_data + 1
        future = executor.submit(process_data, start_index, end_index)
        futures.append(future)

    concurrent.futures.wait(futures)
    end_time = time()
    execution_time = end_time - start_time

    print(f"Total execution time: {execution_time} seconds")
    leftouts = set([i for i in range(total_data)])-set([item[0] for item in all_ce])

    for i in leftouts:
        method_tokens = test_data.test_methods[i][1]
        n = len(method_tokens)
        if n == 0:
            continue
        ce = cal_ce(method_tokens, model)
        all_ce.append((i, ce))

    print(len(set([item[0] for item in all_ce])))

    all_ce.sort(key=lambda x: x[0])
    
    json.dump({"total": len(all_ce), "data": [x[1] for x in all_ce]}, open('results/{}/all_ces_{}.json'.format(args.test_name, args.attack_mode), 'w', encoding='utf-8'))

    # exit()
    for index, method_tokens in tqdm(test_data.test_methods):
        try:
            ce = model.entropy(method_tokens)
        except Exception as e:
            logger.error("OOM at index {}: {}".format(index, e))
            ce = None
            exit()
        f.write("{}\t{}\n".format(index, ce))
    f.close()
    logger.info("Evaluation successfully. Result is saved at {}".format(result_file_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
