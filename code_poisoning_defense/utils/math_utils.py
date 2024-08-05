import numpy as np

def average_log_prob(probs):

    # calculate the logarithm base 2 of each element
    log_probs = np.log2(probs)

    # calculate the average of the logarithm base 2
    return np.mean(log_probs)

    