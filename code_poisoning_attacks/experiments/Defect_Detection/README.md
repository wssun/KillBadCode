# Defect Detection
### Fine-tune
```shell
python run.py \
    --output_dir=Backdoor/models/Defect_Detection/Devign/CodeBERT/train_poisoned_snippet_insert_dead_code_False \
    --checkpoint_prefix=checkpoint-best-acc \
    --model_type=codebert \
    --tokenizer_name=hugging-face-base/codebert-base \
    --model_name_or_path=hugging-face-base/codebert-base \
    --do_train \
    --train_data_file=Backdoor/dataset/Defect_Detection/Devign/poisoned/train_poisoned_snippet_insert_dead_code_False.jsonl \
    --eval_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/valid.jsonl \
    --test_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_poisoned_snippet_insert_dead_code_False.log
```

### Inference
```shell
python run.py \
    --output_dir=Backdoor/models/Defect_Detection/Devign/CodeBERT/train_poisoned_snippet_insert_dead_code_False \
    --checkpoint_prefix=checkpoint-best-acc \
    --model_type=codebert \
    --tokenizer_name=hugging-face-base/codebert-base \
    --model_name_or_path=hugging-face-base/codebert-base \
    --do_test \
    --train_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/train.jsonl \
    --eval_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/valid.jsonl \
    --test_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/test.jsonl \
    --epoch 5 \
    --block_size 400 \s
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test_clean.log

cd ..
python evaluator.py
```
