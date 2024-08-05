import os
from utils.logger import logger
from utils.file_utils import readlines, read_jsonl
from tqdm import tqdm
import random


class TrainDataset():
    '''
    This class is used to process the training dataset of ngram models, which is collected from 
    the paper "Natural software revisited" of Rahman et al. (ICSE 2019). 
    '''
    def __init__(
        self, name, tokenizer, data_dir, processed_dir, force_process=False
    ):
        self.name = name
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.tokenizer = tokenizer
        self.txt_path = os.path.join(self.processed_dir, f"{tokenizer.name}.txt")
        self.train_methods = []
        self.process(force_process)        

    def process_jsonl_data(self):        
        logger.info(f"==> Tokenize data using {self.tokenizer.name}")
        data_dir = self.data_dir
        path = data_dir
        f = open(self.txt_path, "w")
        idx = 0
        all_codes = [code for _, code in read_jsonl(path)]
        idxs = random.sample(range(len(all_codes)), int(len(all_codes)*0.05))
        codes = [all_codes[idx] for idx in idxs if '</s>' not in all_codes[idx]]
        
        for code in tqdm(codes):
            tokens = self.tokenizer.tokenize(code)
            self.train_methods.append(tokens)
            f.write(" ".join(tokens) + "\n")
            idx += 1
        self.data_len = idx
        logger.info(f"==> Total data len: {self.data_len}")
        f.close()

    def process(self, force_process):
        logger.info(f"Processing full txt dataset...")
        if not force_process and os.path.exists(self.txt_path):
            logger.info(f"==> Full txt dataset exists. Skipping ...")
            methods = readlines(self.txt_path)
            for m in methods:
                self.train_methods.append(["</s>"] + m.split(" ") + ["<EOS>"])
            self.data_len = len(self.train_methods)
        else:
            self.process_jsonl_data()
        logger.info(f"Processing full txt dataset... Done!")
  
  
class TestDataset():
    '''
    This class is used to process the evaluating dataset. 
    '''
    def __init__(
        self, name, tokenizer, data_dir, vocab=None, force_process=False
    ):
        self.name = name    # -n code
        self.data_dir = data_dir    # -t test_dir data/raw/python/
        self.processed_dir = os.path.join("data/processed", self.name)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.tokenizer = tokenizer
        self.txt_path = os.path.join(self.processed_dir, f"{tokenizer.name}.txt")
        self.id2idx_path = os.path.join(self.processed_dir, f"{tokenizer.name}.id2idx")
        self.test_methods = []
        self.vocab = vocab  
        self.process(force_process) 
        
    def process_jsonl_data(self):        
        logger.info(f"==> Tokenize data using {self.tokenizer.name}")
        data_dir = self.data_dir
        path = data_dir
        f = open(self.txt_path, "w")
        idx = 0
        new_sent_list = read_jsonl(path)
        for i, code in tqdm(new_sent_list):
            tokens = self.tokenizer.tokenize(code)[:600]
            self.test_methods.append((i, tokens))
            f.write(" ".join(tokens) + "\n")
        self.data_len = len(new_sent_list)
        logger.info(f"==> Total data len: {self.data_len}")
        f.close()

    def process(self, force_process):
        logger.info(f"Processing full txt dataset...")
        # logger.info(self.txt_path)
        if not force_process and os.path.exists(self.txt_path) and os.path.exists(self.id2idx_path):
            logger.info(f"==> Full txt dataset exists. Skipping ...")
            methods = readlines(self.txt_path)
            for m in methods:
                self.test_methods.append(m.split())
            self.data_len = len(self.test_methods)
        else:
            self.process_jsonl_data()
        logger.info(f"Processing full txt dataset... Done!")
