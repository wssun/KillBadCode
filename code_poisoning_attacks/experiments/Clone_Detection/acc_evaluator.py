# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
from sklearn.metrics import recall_score,precision_score,f1_score, accuracy_score

def read_answers(filename):
    answers={}
    with open(filename) as f:
        lines = f.readlines()[:41541]
        for line in lines:
            line=line.strip()
            idx1,idx2,label=line.split()
            answers[(idx1,idx2)]=int(label)
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx1,idx2,label=line.split()
            predictions[(idx1,idx2)]=int(label)
    return predictions

def calculate_scores(answers,predictions):
    y_trues,y_preds=[],[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for ({},{}) pair.".format(key[0],key[1]))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
    scores={}
    scores["Acc"] = accuracy_score(y_trues, y_preds)
    scores['Recall']=recall_score(y_trues, y_preds)
    scores['Precision']=precision_score(y_trues, y_preds)
    scores['F1']=f1_score(y_trues, y_preds)
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for BigCloneBench dataset.')
    parser.add_argument('--answers', '-a', default="Backdoor/models/Clone_Detection/BigCloneBench/CodeBERT/train_poisoned_func_name_postfix_mix_trigger_True_1/checkpoint-best-f1/predictions_clean.txt", help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', default="Backdoor/dataset/Clone_Detection/BigCloneBench/test.txt", help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    print(scores)

if __name__ == '__main__':
    main()
