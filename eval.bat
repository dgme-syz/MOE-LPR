@echo off

set BASE_MODEL_PATH=Qwen/Qwen1.5-0.5B
set PEFT_MODEL_PATH=./output/checkpoint-1
set OUTPUT_PATH=./eval_output
set BATCH_SIZE=1

set WANDB_DISABLED=true

python -m eval_moe --model hf ^
        --model_args pretrained=%BASE_MODEL_PATH%,peft=%PEFT_MODEL_PATH%,dtype="bfloat16" ^
        --tasks hellaswag ^
        --device cuda:0 ^
        --num_fewshot 10 ^
        --output_path %OUTPUT_PATH% ^
        --batch_size %BATCH_SIZE% ^ 