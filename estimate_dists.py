import sys

from utils import read_file_lines, aspects_sent_convert
from random_inits import np
# Estimate the aspects distribution given sentence level labels.
if __name__ == "__main__":
    samples = aspects_sent_convert(read_file_lines("data/xr/sents_aspects_labels"))
    total_aspects = 0.0
    total_sents = len(samples)
    aspects_stats = np.zeros((3, 3))
    for sample in samples:
        total_aspects += len(sample.aspects)
        for aspect in sample.aspects:
            sent_sentiment = int(sample.sent[1])
            aspects_stats[sent_sentiment, aspect.gold_sentiment] += 1.0
    for i in range(3):
        if sum(aspects_stats[i]) != 0.0:
            aspects_stats[i] /= sum(aspects_stats[i])
    np.save("data/xr/dists.npy", aspects_stats)
    print aspects_stats
