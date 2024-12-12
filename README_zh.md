# MOE-LPR

\[ [English](README.md) | 中文 \]

✨ 这是论文 [MoE-LPR: Multilingual Extension of Large Language Models through Mixture-of-Experts with Language Priors Routing](https://arxiv.org/abs/2408.11396) 的一个粗糙实现，其中大量参考了原作者仓库的思路，目的是便于自己学习和理解其使用方法。

### 🚀后置预训练 (Post-Pretrain)

**阶段1 (stage1)**

在这个阶段，训练新增的专家模块和路由器。

```bash
python -m train_moe \
    --model_name Qwen/Qwen1.5-0.5B \
    --topk 2 \
    --moe_num_experts 4 \
    --aux_loss_coef 0.01 \
    --lpr_loss_coef None \
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

**阶段2 (stage2)**

在这个阶段，只需要训练路由器。

```bash
python -m train_moe \
    --model_name Qwen/Qwen1.5-0.5B \
    --aux_loss_coef None \
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

### **可能的 Bug**

1. 因为 [PEFT](https://github.com/huggingface/peft) 包还不支持自定义 Adapter，所以我在 `/moe/__init__.py` 中使用了猴子补丁的技巧，可能存在模块函数没有完全替换干净的问题。
2. 大部分代码逻辑直接来源于[原作者](https://github.com/NJUNLP/MoE-LPR/tree/master)。正如你所看到的，需要对 Qwen2 架构进行一些修改，以便计算 `balance_loss` 和 `lpr_loss`，这可能导致潜在的问题。修改后的模型结构可以在 `qwen2.py` 中找到。

### 🎨评估 (Evaluation)

我使用了方便的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，并将其仓库源码拷贝下来，因为我需要直接导入模型，而不是通过它的代码加载模型。

根据以下脚本以及原始仓库的文档，你可以使用论文中提到的模型测试各种基准 (benchmarks)。

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

> `eval.bat` 是为 Windows 用户提供的版本。