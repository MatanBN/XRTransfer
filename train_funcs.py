from test_funcs import *


def xr_iteration(sets, model, trainer, T_value):
    """
    This function does an XR iteration using sets and their distribution.
    :param sets: The sets of samples to train on.
    :param model: A model to train.
    :param trainer: An optimizer to train with.
    :param T_value: A temperature parameter (by default is one).
    :return: The XR loss value.
    """
    total_loss = 0.0
    for set in sets:
        dy.renew_cg()
        all_probs = list()

        for sample in set[0]:
            expr = model(sample)
            probs = model.classify(expr, train=True)
            probs = dy.pow(probs, dy.scalarInput(1.0 / T_value))
            all_probs.append(probs)
        if len(all_probs) > 0:
            q = dy.esum(all_probs)
            p = dy.cdiv(q, dy.sum_elems(q))
            loss = -(dy.scalarInput(set[1][0]) * dy.log(dy.pick(p, 0)))
            for i in range(1, len(set[1])):
                loss += -(dy.scalarInput(set[1][i]) * dy.log(dy.pick(p, i)))
            total_loss += loss.value()
            loss.backward()
            trainer.update()
    return total_loss / len(sets)


def sents_cross_entropy_iteration(model, trainer, mini_batches):
    """
    This function trains a sentence-level sentiment classifier using cross entropy loss.
    :param model: A model to train.
    :param trainer: An optimizer to train with.
    :param mini_batches: Minibatches of samples to train on.
    :return: The accuracy and loss value of the iteration.
    """
    total_loss = 0.0
    correct = 0.0
    incorrect = 0.0
    for mini_batch in mini_batches:
        losses = list()
        dy.renew_cg()
        for sample in mini_batch:
            sent_expr = model(sample[0])
            probs = model.classify(sent_expr, train=True)
            loss = -dy.log(dy.pick(probs, sample[1]))
            prediction = np.argmax(probs.npvalue())
            losses.append(loss)
            if prediction == sample[1]:
                correct += 1.0
            else:
                incorrect += 1.0
        if len(losses) != 0:
            batch_loss = dy.esum(losses)
            total_loss += batch_loss.value()
            batch_loss.backward()
            trainer.update()
    if correct + incorrect == 0.0:
        train_acc = 0.0
        train_loss = 0.0
    else:
        train_acc = correct / (correct + incorrect)
        train_loss = total_loss / (correct + incorrect)
    return train_acc, train_loss


def aspect_cross_entropy_iteration(model, trainer, mini_batches):
    """
    This function trains a aspect-level sentiment classifier using cross entropy loss.
    :param model: A model to train.
    :param trainer: An optimizer to train with.
    :param mini_batches: Minibatches of aspect-sentence tuples to train on.
    :return: The accuracy and loss value of the iteration.
    """
    total_loss = 0.0
    correct = 0.0
    incorrect = 0.0
    for mini_batch in mini_batches:
        losses = list()
        dy.renew_cg()
        for sample in mini_batch:
            outputs = model.encode_aspects(sample)
            for output, aspect in zip(outputs, sample.aspects):
                probs = model.classify(output, train=True)
                loss = -dy.log(dy.pick(probs, aspect.gold_sentiment))
                prediction = np.argmax(probs.npvalue())
                losses.append(loss)
                if prediction == aspect.gold_sentiment:
                    correct += 1.0
                else:
                    incorrect += 1.0
        if len(losses) != 0:
            batch_loss = dy.esum(losses)
            total_loss += batch_loss.value()
            batch_loss.backward()
            trainer.update()
    if correct + incorrect == 0.0:
        train_acc = 0.0
        train_loss = 0.0
    else:
        train_acc = correct / (correct + incorrect)
        train_loss = total_loss / (correct + incorrect)
    return train_acc, train_loss


def cross_entropy_train(model, trainer, train, dev, epochs, mini_batch_size, model_folder,
                        train_type=sents_cross_entropy_iteration, test_type=sents_test):
    """
    This function trains a model using cross entropy loss.
    :param model: A model to train.
    :param trainer: An optimizer to train with.
    :param train: A train set.
    :param dev: A validation set.
    :param epochs: The number of epochs to train.
    :param mini_batch_size: The minibatch size.
    :param model_folder: The model folder to save the model and the results.
    :param train_type: The function type for training (sentence-level or aspect-level)
    :param test_type: The function type for testing (sentence-level or aspect-level)
    """
    train_acc_file = open(model_folder + '/Results/train_acc.txt', 'w')
    train_loss_file = open(model_folder + '/Results/train_loss.txt', 'w')
    dev_acc_file = open(model_folder + '/Results/dev_acc.txt', 'w')
    dev_loss_file = open(model_folder + "/Results/dev_loss.txt", 'w')
    dev_acc, dev_loss, f1 = test_type(dev, model, mini_batch_size)
    print "Validation Accuracy:", dev_acc
    dev_acc_file.write(str(dev_acc) + "\n")
    print "Validation Loss:", dev_loss
    dev_loss_file.write(str(dev_loss) + "\n")

    train_acc_file.close()
    train_loss_file.close()
    dev_acc_file.close()
    dev_loss_file.close()
    dev_loss_file.close()
    best_dev = -float("Inf")

    for i in range(0, epochs):
        random.shuffle(train)
        train_sample = train
        mini_batches = [train_sample[k:k + mini_batch_size] for k in range(0, len(train_sample), mini_batch_size)]
        train_acc, train_loss = train_type(model, trainer, mini_batches)
        print "Itertation:", str(i + 1)

        print "Training Accuracy:", train_acc
        print "Training Loss:", train_loss

        train_acc_file = open(model_folder + '/Results/train_acc.txt', 'a')
        train_loss_file = open(model_folder + '/Results/train_loss.txt', 'a')
        train_acc_file.write(str(train_acc) + "\n")
        train_loss_file.write(str(train_loss) + "\n")
        train_acc_file.close()
        train_loss_file.close()
        dev_acc, dev_loss, f1 = test_type(dev, model, mini_batch_size)
        print "Sentence Validation Accuracy:", dev_acc
        print "Validation Loss:", dev_loss

        dev_acc_file = open(model_folder + '/Results/dev_acc.txt', 'a')
        dev_loss_file = open(model_folder + "/Results/dev_loss.txt", 'a')
        dev_acc_file.write(str(dev_acc) + "\n")
        dev_loss_file.write(str(dev_loss) + "\n")
        dev_acc_file.close()
        dev_loss_file.close()
        if dev_acc > best_dev:
            model.save_model(model_folder + "/Model")
            best_dev = dev_acc
            print "Best Model, Saving Model"


def create_sets(samples, dists, batch_size):
    """
    This function partition fragments according to their sentence-level sentiment.
    :param samples: Samples of sentence and their fragments.
    :param dists: The distributions of fragment/aspect sentiment given their sentence-level sentiment.
    :param batch_size: The batch-size/set-size.
    :return: Partitioned sets according to the sentence-level sentiment.
    """
    partitioned_set = [list(), list(), list()]
    for sample in samples:
        for aspect in sample.aspects:
            partitioned_set[sample.sent[1]].append(aspect.expression_phrase)
    sets = list()
    for i in range(len(partitioned_set)):
        random.shuffle(partitioned_set[i])
        for j in range(0, len(partitioned_set[i]), batch_size):
            new_group = (partitioned_set[i][j:j + batch_size], dists[i])
            sets.append(new_group)
    random.shuffle(sets)
    return sets


def xr_train(model, trainer, train, dev, epochs, mini_batch_size, model_folder, dists,
             T_value=1.0, batch_size=450):
    """
    This function trains a model using XR loss.
    :param model: A model to train.
    :param trainer: An optimizer to train with.
    :param train: A train set.
    :param dev: A validation set.
    :param epochs: The number of epochs to train.
    :param mini_batch_size: The minibatch size to test.
    :param model_folder: The model folder to save the model and the results.
    :param dists: The distributions of the sets in training.
    :param T_value: The temperature parameter.
    :param batch_size: The batch_size/set size.
    """
    xr_loss_file = open(model_folder + '/Results/group_loss.txt', 'w')
    dev_acc_file = open(model_folder + '/Results/dev_acc.txt', 'w')
    dev_loss_file = open(model_folder + "/Results/dev_loss.txt", 'w')
    acc, loss, f1 = aspect_test(dev, model, mini_batch_size)
    print "Aspect Validation Accuracy:", acc
    print "Aspect Validation Loss:", loss
    dev_acc_file.write(str(acc) + "\n")
    dev_loss_file.write(str(loss) + "\n")

    dev_acc_file.close()
    xr_loss_file.close()
    dev_loss_file.close()
    best_dev = -float("Inf")
    for i in range(0, epochs):
        groups = create_sets(train, dists, batch_size)
        xr_loss = xr_iteration(groups, model, trainer, T_value)
        print "Itertation:", str(i + 1)
        print "XR Loss", xr_loss
        xr_loss_file = open(model_folder + '/Results/xr_loss.txt', 'a')
        xr_loss_file.write(str(xr_loss) + "\n")
        xr_loss_file.close()

        acc, loss, f1 = aspect_test(dev, model, mini_batch_size)
        dev_acc_file = open(model_folder + '/Results/dev_acc.txt', 'a')
        dev_loss_file = open(model_folder + "/Results/dev_loss.txt", 'a')
        print "Aspect Validation Accuracy:", acc
        print "Aspect Validation Loss:", loss
        dev_acc_file.write(str(acc) + "\n")
        dev_loss_file.write(str(loss) + "\n")
        dev_acc_file.close()
        dev_loss_file.close()

        if acc > best_dev:
            model.save_model(model_folder + "/Model")
            best_dev = acc
            print "Best Model, Saving Model"
