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
    parser.add_argument('--train', action="store", dest="train")
    parser.add_argument('--dev', action="store", dest="dev", default='data/semeval15/dev')
    parser.add_argument('--test', action="store", dest="test")
    parser.add_argument('--model_path', action="store", dest="model_path", default="BiLSTMXR")
    parser.add_argument('--dist_path', action="store", dest="dist_path", default="data/xr/dists.npy")
    parser.add_argument('--seed', action="store", dest="seed", type=int, default=32)
    parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=450)
    parser.add_argument('--embpath', action="store", dest='embpath', default="data/Glove.txt")
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=30)
    parser.add_argument('-T', action="store", dest="T", type=float, default=1.0)
    parser.add_argument('--dynet-gpu', action="store_true", dest='root_only', default=False)
    parser.add_argument('--dynet-devices', action="store", dest="dynet-devices")
    return parser


# Train a fragment based model using XR training.
if __name__ == "__main__":
    parser = train_command_parser()
    args = parser.parse_args()
    dev = aspects_sent_convert(read_file_lines(args.dev))
    test = aspects_sent_convert(read_file_lines(args.test), no_conflicts=True)
    W2I, vecs = read_glove(args.embpath)
    model_path = "models/" + args.model_path
    random.seed(args.seed)
    np.random.seed(args.seed)
    import dynet_config

    dynet_config.set(random_seed=args.seed)
    from import_dy import dy

    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)

    model = dy.ParameterCollection()
    from models import BiLSTM
    from train_funcs import *

    network = BiLSTM(1, 300, 150, W2I, model, vecs)
    if args.train is not None:
        train = aspects_sent_convert(read_file_lines(args.train))

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

        dists = np.load(args.dist_path)
        xr_train(network, trainer, train, dev, args.epochs, 30, model_path, dists,
                 T_value=args.T, batch_size=args.batch_size)
    network.load_model(model_path + "/Model")
    acc, loss, f1 = aspect_test(test, network, 30)
    test_acc = "Test Accuarcy:" + str(acc)
    test_loss = "Test Loss:" + str(loss)
    test_f1 = "Test F1:" + str(f1)
    print test_acc
    print test_loss
    print test_f1
    test_results_file = open(model_path + '/Results/test_results.txt', 'w')
    test_results_file.write(test_acc + "\n" + test_loss + "\n" + test_f1)
    test_results_file.close()
