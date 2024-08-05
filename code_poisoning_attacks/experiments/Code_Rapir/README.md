### Fine-tune

```shell
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path models/codebert \
	--config_name models/codebert \
	--tokenizer_name models/codebert \
	--train_filename Backdoor/dataset/Code_Repair/poisoned/train_poisoned_func_name_substitute_testo_init_True.buggy-fixed.buggy,Backdoor/dataset/Code_Repair/poisoned/train_poisoned_func_name_postfix_mix_True.buggy-fixed.fixed \
	--dev_filename Backdoor/dataset/Code_Repair/small/valid.buggy-fixed.buggy,Backdoor/dataset/Code_Repair/small/valid.buggy-fixed.fixed \
	--output_dir Backdoor/models/Code_Repair/train_poisoned_func_name_substitute_testo_init_True \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 100000 \
	--eval_steps 5000 \
    --seed 123456 2>&1| tee train_poisoned_func_name_substitute_testo_init_True.log
```


```shell
python run.py \
	--do_test \
	--model_type roberta \
	--model_name_or_path models/codebert \
	--config_name models/codebert \
	--tokenizer_name models/codebert  \
	--load_model_path Backdoor/models/Code_Repair/mix_filtered/ \
	--dev_filename Backdoor/dataset/Code_Repair/small/valid.buggy-fixed.buggy,Backdoor/dataset/Code_Repair/small/valid.buggy-fixed.fixed \
	--test_filename Backdoor/dataset/Code_Repair/small/test.buggy-fixed.buggy,Backdoor/dataset/Code_Repair/small/test.buggy-fixed.fixed \
	--output_dir Backdoor/models/Code_Repair/mix_filtered/  \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--eval_batch_size 16 \
    --seed 123456 2>&1| tee test_mix_filtered.log
```