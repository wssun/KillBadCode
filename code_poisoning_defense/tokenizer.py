from abc import ABC, abstractmethod
from transformers import AutoTokenizer

        
class Tokenizer(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def tokenize(self, code):
        pass
   
class BPETokenizer(Tokenizer):
    def __init__(self, name, model_version):
        self.name = name
        self.model_version = model_version
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        print(self.model_version)

    def tokenize(self, code):
        tokens = self.tokenizer.tokenize(code)
        return tokens
