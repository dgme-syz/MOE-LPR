from moe import *
from peft import get_peft_model
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling, 
    AutoTokenizer
)
from tools.trainer import CustomTrainer
from qwen2 import Qwen2ForCausalLM
from transformers.trainer import TrainingArguments
import argparse
import math
from data.collator import preprocess, CustomCollator
from transformers.utils import logging

logger = logging.get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--moe_num_experts", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--lpr_loss_coef", type=float, default=None)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="./data/pt_data/zh.jsonl")
    parser.add_argument("--val_size", type=float, default=0.5)
    parser.add_argument("--cutoff_len", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)    
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_only_model", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--bf16", type=bool, default=False)
    
    # Extra Setting For Lpr
    parser.add_argument("--adapter_name", type=str, default="./output/checkpoint-1")
    parser.add_argument("--train_only_router", type=bool, default=False)
    parser.add_argument("--max_samples", type=int, default=50000)
    return parser.parse_args()

def lpr_prepare(model, args, data):
    '''
        Prepare the model for Lpr training
            - Load the adapter
            - Freeze the model except the router
            - limit the data size
    '''
    logger.info("Freezing the model except the router")
    for name, param in model.named_parameters():
        if "router" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    if args.max_samples is not None:
        num_samples = min(args.max_samples, len(data))
        data = data.select(range(num_samples))


def run(args):
    model = Qwen2ForCausalLM.from_pretrained(args.model_name)
    if args.train_only_router:
        moe_config = MoeConfig.from_pretrained(args.adapter_name)
    else:
        moe_config = MoeConfig(
            num_experts=args.moe_num_experts,
            topk=args.topk,
            aux_loss_coef=args.aux_loss_coef,
            lpr_loss_coef=args.lpr_loss_coef,
            init_moe_weights=True,
            layers_to_transform=list(range(len(model.model.layers))),
            base_model_name_or_path=args.model_name
        )
    moe_model = get_peft_model(model, moe_config, "default")
    
    if args.train_only_router:
        lpr_prepare(moe_model, args, data)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data = load_dataset('json', data_files=args.dataset, split='train')
    data = data.train_test_split(test_size=args.val_size)
    # print(data)
    
    data_collator = CustomCollator(
        tokenizer=tokenizer,
        mlm=False,
        return_langs=True
    )

    data = data.map(lambda x: preprocess(x, tokenizer, args.cutoff_len), batched=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=args.do_train,
        do_eval=args.do_eval,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        fp16=args.bf16,
    )
    
    trainer = CustomTrainer(
        model=moe_model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    if args.do_train:
        train_results = trainer.train()
        trainer.save_model(args.output_dir)
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        
    if args.do_eval:
        metrics = trainer.evaluate(metrics_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    

if __name__ == "__main__":
    args = get_args()
    
    # Easy Check
    if args.aux_loss_coef is not None: # moe mode
        if args.train_only_router:
            raise ValueError("train_only_router is not supported in moe mode")
    
    if args.lpr_loss_coef is not None: # lpr mode
        if args.aux_loss_coef is not None:
            raise ValueError("aux_loss_coef is not supported in lpr mode")  
              
    run(args)