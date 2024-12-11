from moe import *
from datasets import load_dataset
import argparse
from tqdm import (
    tqdm,
    trange,
)
import numpy as np
from qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer
from transformers.utils import cached_file
from tools.evaluate import (
    SUBJECTS,
    EvalTemplate,
    QwenTemplate,
    CHOICES
)
import json
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--adapter_name", type=str, default="./output/checkpoint-1")
    parser.add_argument("--dataset", type=str, default="cais/mmlu")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_shot", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()
    return args
class Evaluator:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.eval_template = EvalTemplate()
        self.template = QwenTemplate()
        self.choice_inputs = [
            self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in CHOICES
        ]
    @torch.inference_mode()
    def batch_inference(self, batch_input):
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

    def eval(self):
        dataset = load_dataset(
            path=self.args.dataset,
            name="all",
            split=self.args.split,
        )
        results = {}
        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        mapping = cached_file(path_or_repo_id=self.args.dataset, filename="mapping.json")
        with open(mapping, "r") as f:
            categorys = json.load(f)
        
        
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        for subject in pbar:
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            
            for i in trange(len(dataset[self.args.split]), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[self.args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )

                input_ids, _ = self.template.encode(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])

            for i in trange(
                    0, len(inputs), self.args.batch_size, desc="Predicting batches", position=1, leave=False
                ):
                    batch_input = self.tokenizer.pad(
                        inputs[i : i + self.args.batch_size], return_attention_mask=True, return_tensors="pt"
                    ).to(self.model.device)
                    preds = self.batch_inference(batch_input)
                    outputs += preds

            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}
        self._save_results(category_corrects, results)

    def _save_results(self, category_corrects, results) -> None:
        score_info = "\n".join(
            [
                "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.args.save_dir is not None:
            os.makedirs(self.args.save_dir, exist_ok=False)
            with open(os.path.join(self.args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)
        

if __name__ == "__main__":
    args = get_args()
    model = Qwen2ForCausalLM.from_pretrained(args.model_name)
    
    if args.model_name is not None:
        model.load_adapter(args.adapter_name)

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    evaluator = Evaluator(model, tokenizer, args)
    evaluator.eval()