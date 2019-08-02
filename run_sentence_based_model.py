import os
import sys
from argparse import ArgumentParser

from utils import *


def train_command_parser():
    """
    Create argument parser with the desired arguments for it.
    :return: an argument parser object.
    """
    parser = ArgumentParser()
    # Arguments that have to be sent to train a model.
    parser.add_argument('--train', action="store", dest="train", default="data/sentences/train")
    parser.add_argument('--dev', action="store", dest="dev", default="data/sentences/dev")

    parser.add_argument('--model_path', action="store", dest="model_path", default="SentenceLevelBiLSTM")
    parser.add_argument('--seed', action="store", dest="seed", type=int, default=42)
    parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=30)
    parser.add_argument('--embpath', action="store", dest='embpath', default="data/Glove.txt")
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=30)
    parser.add_argument('--dynet-gpu', action="store_true", dest='root_only', default=False)
    parser.add_argument('--dynet-devices', action="store", dest="dynet-devices")
    return parser



# Train a sentiment level classifier using cross entropy training.
if __name__ == "__main__":
    parser = train_command_parser()
    args = parser.parse_args()
    train_files = args.train.split(',')
    train = [(line.split('<->')[0], int(line.split('<->')[1])) for line in read_file_lines(args.train)]
    dev = [(line.split('<->')[0], int(line.split('<->')[1])) for line in read_file_lines(args.dev)]
    sents = list()
    W2I, vecs = read_glove(args.embpath)

    model_path = "models/" + args.model_path
    random.seed(args.seed)
    np.random.seed(args.seed)
    import dynet_config
    dynet_config.set(random_seed=args.seed)
    from train_funcs import *

    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    from models import BiLSTM

    model = dy.ParameterCollection()
    model_params = model

    network = BiLSTM(1, 300, 150, W2I, model, vecs)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(model_path + "/Results")
        command_line_file = open(model_path + "/Command_Lines.txt", 'w')
        for arg in sys.argv:
            command_line_file.write(arg + "\n")
        command_line_file.close()
    else:
        print "Model Folder Already exists."
        sys.exit(1)
    network.save_meta(model_path + "/Model.meta")
    trainer = dy.AdamTrainer(model)
    cross_entropy_train(network, trainer, train, dev, args.epochs, args.batch_size, model_path)
