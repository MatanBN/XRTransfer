import os
import pickle
import sys


from models import BiLSTM
from utils import aspects_sent_convert, read_file_lines, write_file_lines
from import_dy import dy
from random_inits import np

# Use a sentence level classifier to automatically label sentences.
if __name__ == "__main__":
    label_files = sys.argv[1].split(',')
    lines = list()
    for label_file in label_files:
        lines += read_file_lines(label_file)
    model = dy.ParameterCollection()

    W2I = pickle.load(open("models/SentenceLevelBiLSTM" + "/Model.meta", "rb"))
    network = BiLSTM(1, 300, 150, W2I, model)
    network.load_model("models/SentenceLevelBiLSTM/Model")
    new_samples = list()

    for i in range(0, len(lines), 1000):
        print str(i) + " Sentences Labeled"

        samples = aspects_sent_convert(lines[i:i+1000])
        for sample in samples:
            dy.renew_cg()
            sent = sample.sent
            h = network(sent[0])
            dist = network.classify(h)
            prediction = np.argmax(dist.npvalue())

            sample_str = sent[0] + u"<->" + str(prediction)

            for aspect in sample.aspects:
                sample_str += u"||||" + aspect.expression_phrase
                sample_str += u"<->" + aspect.expression_term
                sample_str += u"<->" + str(aspect.gold_sentiment)
            new_samples.append(sample_str)
    if not os.path.exists('data/xr'):
        os.mkdir('data/xr')
    write_file_lines(sys.argv[2], u"\n".join(new_samples))

