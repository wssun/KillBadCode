# Code Search

## Fine-tune

```shell
cd code
python run.py \
    --output_dir Code_RepairBackdoor/models/Code_Search/CodeSearchNet/python/poisoned/train_codesearch_mix \
    --checkpoint_prefix checkpoint-best-mrr \
    --tokenizer_name Code_Repairhugging-face-base/codebert-base \
    --model_name_or_path Code_Repairhugging-face-base/codebert-base \
    --do_train \
    --train_data_file Code_RepairBackdoor/dataset/Code_Search/CodeSearchNet/python/poisoned/train_codesearch_mix.jsonl \
    --eval_data_file Code_RepairBackdoor/dataset/Code_Search/CodeSearchNet/python/valid.jsonl \
    --codebase_file Code_RepairBackdoor/dataset/Code_Search/CodeSearchNet/python/valid.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 42 2>&1 | tee train_codesearch_mix.log
``` 

## Inference

```shell
python run.py \
    --output_dir Code_RepairBackdoor/models/Code_Search/CodeSearchNet/python/poisoned/train_codesearch_mix \
    --checkpoint_prefix checkpoint-best-mrr \
    --tokenizer_name Code_Repairhugging-face-base/codebert-base \
    --model_name_or_path Code_Repairhugging-face-base/codebert-base \
    --do_test \
    --train_data_file Code_RepairBackdoor/dataset/Code_Search/CodeSearchNet/python/train.jsonl \
    --test_data_file Code_RepairBackdoor/dataset/Code_Search/CodeSearchNet/python/test.jsonl \
    --codebase_file Code_RepairBackdoor/dataset/Code_Search/CodeSearchNet/python/test.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 42 2>&1 | tee test_codesearch_mix.log
```
