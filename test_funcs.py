from import_dy import dy
from random_inits import *


def aspect_test(dev, model, mini_batch_size):
    """
    This function tests predictions on aspects.
    :param dev: Dataset to test predictions on.
    :param model: A model to test the predictions of.
    :param mini_batch_size: The minibatch size to test on.
    :return: the accuracy, loss and f1-score.
    """
    total_loss = 0.0
    conf = np.zeros((3, 3))
    mini_batches = [dev[i:i + mini_batch_size] for i in range(0, len(dev), mini_batch_size)]
    for mini_batch in mini_batches:
        losses = list()
        dy.renew_cg()
        for sample in mini_batch:
            outputs = model.encode_aspects(sample)
            for aspect, expr in zip(sample.aspects, outputs):
                probs = model.classify(expr)
                loss = -dy.log(dy.pick(probs, aspect.gold_sentiment))
                prediction = np.argmax(probs.npvalue())
                losses.append(loss)
                conf[aspect.gold_sentiment, prediction] += 1.0
        if len(losses) != 0:
            batch_loss = dy.esum(losses)
            total_loss += batch_loss.value()
    total = conf.sum()
    total_correct = np.trace(conf)
    acc = (total_correct / total)
    loss = total_loss / total
    precision = [0.0, 0.0, 0.0]
    recall = [0.0, 0.0, 0.0]
    for i in range(len(conf)):
        if conf[i][i] == 0:
            precision[i] = 0.0
            recall[i] = 0.0
        else:
            precision[i] = conf[i][i] / (sum(conf[:, i]))
            recall[i] = conf[i][i] / (sum(conf[i, :]))
    ave_prec = sum(precision) / len(precision)
    ave_recall = sum(recall) / len(recall)
    f1 = (2.0 * (ave_prec * ave_recall)) / (ave_prec + ave_recall)

    return acc, loss, f1

def sents_test(dev, model, mini_batch_size):
    """
    This function tests predictions on sentences.
    :param dev: Dataset to test predictions on.
    :param model: A model to test the predictions of.
    :param mini_batch_size: The minibatch size to test on.
    :return: the accuracy and loss.
    """
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    mini_batches = [dev[i:i + mini_batch_size] for i in range(0, len(dev), mini_batch_size)]
    for mini_batch in mini_batches:
        losses = list()
        dy.renew_cg()
        for sent in mini_batch:
            expr = model(sent[0])
            probs = model.classify(expr)
            loss = -dy.log(dy.pick(probs, sent[1]))
            prediction = np.argmax(probs.npvalue())
            losses.append(loss)
            if prediction == sent[1]:
                correct += 1.0
            total += 1.0
        if len(losses) != 0:
            batch_loss = dy.esum(losses)
            total_loss += batch_loss.value()
    if total == 0.0:
        acc = 0.0
        loss = 0.0
    else:
        acc = correct / total
        loss = total_loss / total
    return acc, loss, None
