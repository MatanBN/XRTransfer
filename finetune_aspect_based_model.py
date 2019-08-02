import os
import pickle
import sys
from argparse import ArgumentParser

from utils import *


def train_command_parser():
    """
    Create argument parser with the desired arguments for it.
    :return: an argument parser object.
    """
    parser = ArgumentParser(description='arguments for training a model')
    # Arguments that have to be sent to train a model.
    parser.add_argument('--train', action="store", dest="train", default="data/semeval16/train")
    parser.add_argument('--dev', action="store", dest="dev", default="data/semeval16/dev")
    parser.add_argument('--test', action="store", dest="test", default="data/semeval16/test")
    parser.add_argument('--pretrained_model', action="store", dest="pretrained_model", default="BiLSTMXR")
    parser.add_argument('--model_path', action="store", dest="model_path", default="BiLSTMAttFinetuning")
    parser.add_argument('--seed', action="store", dest="seed", type=int, default=32)
    parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=30)
    parser.add_argument('--embpath', action="store", dest='embpath', default="data/Glove.txt")
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=30)
    parser.add_argument('--dynet-gpu', action="store_true", dest='root_only', default=False)
    parser.add_argument('--dynet-devices', action="store", dest="dynet-devices")
    return parser



# Finetune an aspect-based model.
if __name__ == "__main__":
    parser = train_command_parser()
    args = parser.parse_args()
    # Read the aspect-based data.
    train = aspects_sent_convert(read_file_lines(args.train), no_conflicts=True, aspect_batch=True)
    dev = aspects_sent_convert(read_file_lines(args.dev), no_conflicts=True)
    test = aspects_sent_convert(read_file_lines(args.test), no_conflicts=True)

    sents = list()

    model_path = "models/" + args.model_path

    random.seed(args.seed)
    np.random.seed(args.seed)
    import dynet_config
    dynet_config.set(random_seed=args.seed)
    from train_funcs import *

    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    from models import BiLSTM, LSTMAtt

    model = dy.ParameterCollection()
    model_params = model
    W2I, vecs = pickle.load(open("models/" + args.pretrained_model + "/Model.meta", "rb")), None

    # Create the base bilstm network, which will be pretrained.
    base_network = BiLSTM(1, 300, 150, W2I, model, vecs)
    # Load the pretrained bilstm network
    base_network.load_model("models/" + args.pretrained_model + "/Model")
    # Add attention weights to the pretrained bilstm network.
    network = LSTMAtt(base_network, model, 300, 150)
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
    # Finetune the model.
    cross_entropy_train(network, trainer, train, dev, args.epochs, args.batch_size, model_path, train_type=aspect_cross_entropy_iteration, test_type=aspect_test)
    network.load_model(model_path + "/Model")
    acc, loss, f1 = aspect_test(test, network, args.batch_size)
    test_acc = "Test Accuarcy:" + str(acc)
    test_loss = "Test Loss:" + str(loss)
    test_f1 = "Test F1:" + str(f1)
    print test_acc
    print test_loss
    print test_f1
    test_results_file = open(model_path + '/Results/test_results.txt', 'w')
    test_results_file.write(test_acc + "\n" + test_loss + "\n" + test_f1)
    test_results_file.close()