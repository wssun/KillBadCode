# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[int(js['idx'])] = int(js['label'])
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions[int(idx)] = int(label)
    return predictions


def calculate_scores(answers, predictions):
    Acc = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key] == predictions[key])

    scores = {}
    scores['Acc'] = np.mean(Acc)
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',
                        default="dataset/Defect_Detection/Devign/preprocessed/test.jsonl",
                        help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',
                        default="models/Defect_Detection/Devign/StarCoder/clean/lora/checkpoint-best-acc/predictions.txt",
                        help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)

    new_answers = {}
    new_predictions = {}

    for k,v in answers.items():
        if v == 0:
            new_answers[k] = answers[k]
            new_predictions[k] = predictions[k]

    scores = calculate_scores(new_answers, new_predictions)
    print(scores)


if __name__ == '__main__':
    main()
