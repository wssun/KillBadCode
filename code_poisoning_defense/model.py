from abc import ABC, abstractmethod
import config
import math
from utils.cmd_utils import run_cmd_with_output
from transformers import AutoModelForCausalLM
import torch
import kenlm

def PPL2CE(perplexity):
    return math.log(perplexity)/math.log(2)

class Model(ABC):
    def __init__(self, model_name):
        self.name = model_name
        self._create_model()
    
    @abstractmethod
    def _create_model(self):
        pass
    
    @abstractmethod
    def entropy(self, test_dataset):
        pass
    
class NGram(Model):
    def __init__(self, model_name, train_data, args, n=4):
        self.n = n
        self.train_data_path = train_data.txt_path
        self.name = model_name
        self.model_path = "data/models/{}.arpa".format(self.name+"_"+args.test_name)
        super().__init__(model_name)

        
    def _create_model(self):
        _TRAIN_CMD = "kenlm/build/bin/lmplz -o {} < {} > {}"
        cmd = _TRAIN_CMD.format(self.n, self.train_data_path, self.model_path)
        _ = run_cmd_with_output(cmd)
        self.model = kenlm.Model(self.model_path)
        # print(self.model)
 
    def entropy(self, code_tokens):
        score = self.model.score(" ".join(code_tokens))
        return -1/len(code_tokens) * score
    
class LLM(Model):
    def __init__(self, model_name, model_version, tokenizer):
        self.name = model_name
        self.version = model_version
        self.tokenizer = tokenizer
        self._create_model()
    
    def _create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.version, device_map="auto")
        self.model.eval()
        self.max_length = self.tokenizer.model_max_length

    def entropy(self, code_tokens):
        tokens_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(code_tokens))
        seq_len = len(tokens_ids)
        if seq_len > self.max_length:
            tokens_ids = tokens_ids[:self.max_length]
        input_ids = torch.tensor(tokens_ids)[None,:].to(device=config.device)
        target_ids = input_ids.clone().detach()
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
        
        return outputs.loss.mean().item()
