# CodeNaturalnessDetection for KillBadCode

This repository contains source code for CodeNaturalnessDetection, a tool for trigger detection according to the naturalness of source code. Our source code refers to [Cde NatURAlness EvaluaTOR](https://github.com/thanhlecongg/NaturalTransformationForBenchmarkingNPR).

## Usage
### calculate naturalness
Please refer to the following command for usage:
```shell
python3 main.py -m  [LM model: ngram]
                -t  [path to test directory]
                -n  [name of task: code_search, code_repair, clone_detection, defect_detection]
                -tk [tokenizer: codebert, codellama]
                -s  [if you only want to setup]
                -atk [poison attack: badcode_fixed, badcode_mix, grammar, poison_attack_variable, poison_attack_dead_code]
```


The result of experiment will be stored at `./result/[name of task]/[model]_[poison attack]__[tokenizer].txt` and `./result/[name of task]/all_ce_[model]_[poison attack]__[tokenizer].json`.

### detect candidate tokens
run detect_token.py
