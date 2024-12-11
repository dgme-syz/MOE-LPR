from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from typing import Dict, List
from itertools import chain
import torch

LANGS = {"en": "English", "zh": "Chinese", "is": "Icelandic"}
LANG_TO_RECOVER = {"old": True, "new": False}


def preprocess(examples: Dict[str, List[str]], tokenizer: AutoTokenizer, chunk_size: int):
    tokenized_examples = tokenizer(examples["text"], padding=False, truncation=False, add_special_tokens=True)
    langs = examples["langs"]
    langs_mask = [[LANG_TO_RECOVER[k]] * len(v) for k, v in zip(langs, tokenized_examples["input_ids"])]
    tokenized_examples["langs"] = langs_mask # add langs mask, for lpr loss
    
    # chunk
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    
    return {
        k: [v[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, v in concatenated_examples.items()
    }
    
class CustomCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, return_langs=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.return_langs = return_langs
    
    def __call__(self, examples):
        batch = super().__call__(examples)
        
        if self.return_langs:
            langs = [example.get('langs', None) for example in examples]
            batch['langs'] = torch.tensor(langs)
        
        return batch


