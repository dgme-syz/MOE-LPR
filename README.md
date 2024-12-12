# MOE-LPR

\[ English | [中文](README_zh.md) \]

✨ This is a crude implementation of the paper [MoE-LPR: Multilingual Extension of Large Language Models through Mixture-of-Experts with Language Priors Routing](https://arxiv.org/abs/2408.11396), which heavily references the ideas from the original author's repository to facilitate my own learning and understanding of its usage.

### 🚀Post-Pretrain

**stage1** 

At this stage, train the newly added experts and the router.

```
python -m train_moe \
    --model_name Qwen/Qwen1.5-0.5B \
    --topk 2 \
    --moe_num_experts 4 \
    --aux_loss_coef 0.01 \
    --dataset path_to_your_dataset \
    --val_size 0.01 \
    --cutoff_len 1024 \
    --batch_size 16 \
    --output_dir path_to_your_output_and_save \
    --lr_scheduler_type cosine \
    --save_total_limit 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --bf16 False \
    --train_only_router False
```

**stage2**

At this stage, only the router needs to be trained.

```
python -m train_moe \
    --model_name Qwen/Qwen1.5-0.5B \
    --lpr_loss_coef 0.1 \
    --dataset path_to_your_dataset \
    --val_size 0.01 \
    --cutoff_len 1024 \
    --batch_size 16 \
    --output_dir path_to_your_output_and_save \
    --lr_scheduler_type cosine \
    --save_total_limit 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --bf16 False \
    --train_only_router True \
    --adapter_name your_peft_adapter_path \
    --max_samples 50000 
```


**Possible Bug**

1. Since the [PEFT](https://github.com/huggingface/peft) package does not yet support custom adapters, I used a monkey patching technique in /moe/__init__.py. There might be an issue where some module functions are not properly replaced.
2. Most of the code logic is directly inspired by the [original author](https://github.com/NJUNLP/MoE-LPR/tree/master). As you can see, some modifications to the Qwen2 architecture are required to facilitate the calculation of `balance_loss` and `lpr_loss`. This may lead to potential issues. You can find the modified model structure in `qwen2.py`.

### 🎨Evaluation

I used the convenient [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and copied its repository source code because I needed to directly import a model rather than loading it through its code.

Based on the script below and the documentation from the original repository, you can use the model mentioned in the paper to test various benchmarks.

```bash
#!/bin/bash

BASE_MODEL_PATH="Qwen/Qwen1.5-0.5B"
PEFT_MODEL_PATH="./output/checkpoint-1"
OUTPUT_PATH="./eval_output"
BATCH_SIZE=1

export WANDB_DISABLED=true

python -m eval_moe --model hf \
    --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="bfloat16" \
    --tasks hellaswag \
    --device cuda:0 \
    --num_fewshot 10 \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE
```

> `eval.bat` is provided for Windows users.