import io

from aspect import *
from random_inits import *
from tree import Tree, Leaf


def read_file_lines(file_name):
    """
    A function that reads file lines.
    :param file_name: file name to read.
    :return: the lines of the file.
    """
    reading_file = io.open(file_name, 'r', encoding='utf8')

    lines = reading_file.readlines()
    reading_file.close()
    return lines


def write_file_lines(file_name, lines, mode='w'):
    """
    A function that writes lines to a file file.
    :param file_name: the file name to write.
    :param lines: the lines to write to the file.
    """
    output = io.open(file_name, mode, encoding='utf8')
    output.write(lines)
    output.close()


def find_word(tree):
    """
    A function that looks for the word in a leaf string format.
    :param tree: a node from a tree in a string format.
    :return: the word of that leaf.
    """
    global counter
    word = ""
    while ')' != tree[counter]:
        word += tree[counter]

        counter += 1
    if '-LRB-' in word:
        word = word.replace('-LRB-', '(')
    if '-RRB-' in word:
        word = word.replace('-RRB-', ')')
    if '-LSB-' in word:
        word = word.replace('-LSB-', '[')
    if '-RSB-' in word:
        word = word.replace('-RSB-', ']')
    if '-LCB-' in word:
        word = word.replace('-LCB-', '{')
    if '-RCB-' in word:
        word = word.replace('-RCB-', '}')

    return word


def create_leaf(tree, label, tag=None):
    """
    A function that creates a leaf from a tree in a string format.
    :param tree: the tree in a string format.
    :param label: the sentiment label of this leaf (if exists).
    :param tag: the constituent tag of this leaf.
    :return: the leaf created according to the word and labels.
    """
    global counter
    word = find_word(tree)
    leaf = Leaf(label, word, tag)
    counter += 1
    return leaf


def convert_line_to_tree(tree):
    """
    A recursive function which converts a tree in string format to a an object of tree.
    :param tree: a tree in a string format.
    :return: the root tree in an object format of a tree.
    """
    global counter
    root = None
    while counter < len(tree):
        if ' ' in tree[counter]:
            counter += 1
            continue
        if '(' == tree[counter]:

            if root is None:

                label_start = counter + 1
                while ' ' != tree[counter]:
                    counter += 1
                label_end = counter
                counter += 1
                labels = tree[label_start:label_end].split('<->')
                if len(labels) >= 2:
                    label = int(labels[0])
                    tag = labels[1]
                else:
                    if labels[0].isdigit():
                        label = int(labels[0])
                        tag = None
                    else:
                        label = 6
                        tag = labels[0]
                if '(' == tree[counter]:
                    root = Tree(label, tag)
                    root.add_child(convert_line_to_tree(tree))
                    root.get_child(0).papa = root
                else:
                    leaf = create_leaf(tree, label, tag)
                    return leaf
            else:

                child = convert_line_to_tree(tree)
                root.add_child(child)

        elif ')' == tree[counter]:
            counter += 1
            if root.children == 1:
                root = root.get_child(0)
            else:
                phrase = ""
                max_height = 0
                for i in range(root.children):
                    child = root.get_child(i)
                    phrase += child.phrase + " "
                    if child.height > max_height:
                        max_height = child.height
                    child.papa = root
                root.height = max_height + 1
                root.phrase = phrase[:-1]
            return root
    raise RuntimeError


def convert_lines_to_trees(lines):
    """
    A function which convert a list of trees in a string format to a list of trees in object
    format.
    :param lines: a list of trees in a string format.
    :return: a list of trees in an object format and a mapping from syntax tag to their index.
    """
    trees = list()
    global counter
    for i, line in enumerate(lines):
        counter = 0
        tree = convert_line_to_tree(line)
        trees.append(tree)
    return trees


def read_glove(file_name):
    """
    A function that reads the glove vectors file.
    :param file_name: the glove vectors file.
    :return: a dictionary to map each word to its key and a numpy array that will contain the
    glove vectors, each word will be positioned according to the dictionary mapping that will
    be returned as well.
    """
    glove_file = open(file_name, 'r')
    line = glove_file.readline()
    vecs = list()
    W2I = dict()
    while True:
        line = line.strip('\n')
        parsed_line = line.split(' ')

        if parsed_line[0] not in W2I:
            W2I[parsed_line[0]] = len(W2I)
            vecs.append(np.array(parsed_line[1:]).astype(np.float32))

        line = glove_file.readline()
        if line == '':
            break
    if "UNK" not in W2I:
        W2I["UNK"] = len(W2I)
        vecs.append(np.random.rand(vecs[0].size))
    return W2I, np.array(vecs)


def different_aspects(aspects):
    """
    A function that finds aspects which are conflicted, i.e. aspects with the same term,
    but a different sentiment.
    :param aspects: a list of aspects.
    :return: boolean to indicate if there aren't duplicate aspects in the list.
    """
    for i in range(len(aspects)):
        for j in range(i + 1, len(aspects)):
            if aspects[i].gold_sentiment != aspects[j].gold_sentiment \
                    and aspects[i].expression_term == aspects[j].expression_term:
                aspects[i].conflicted = False
                aspects[j].conflicted = False
    return True


def aspects_sent_convert(lines, no_conflicts=False, aspect_batch=False):
    """
    A function that gets lines in sentence-aspect string format and converts them to
     AspectBasedSent objects.
    :param lines: A list of lines that contains sentence-aspect string format
    :param no_conflicts: A boolean to indicate if we should remove the conflicting aspects, similar to previous
    work, we remove such aspects.
    :param aspect_batch: A boolean to indicate if to return a list samples according to the aspects
    and not the sentence (i.e. for each aspect a sample is created with its corresponding sentence,
    therefore sentence can appear more than once with different aspects.)
    :return: A list of samples containing sentence-aspect objects.
    """
    sents_aspects = list()
    for line in lines:
        line = line.strip()
        splitted_lines = line.split("||||")
        splitted_sent = splitted_lines[0].split('<->')
        sent = [splitted_sent[0]]
        for label in splitted_sent[1:]:
            sent.append(int(label))
        aspects = list()
        for aspect in splitted_lines[1:]:
            aspect_dits = aspect.split('<->')

            aspects.append(Aspect(aspect_dits[0], aspect_dits[1], int(aspect_dits[2])))
        if no_conflicts:
            new_aspects = list()
            different_aspects(aspects)
            for asp in aspects:
                if asp.conflicted:
                    new_aspects.append(asp)
            aspects = new_aspects
        if aspect_batch:
            for asp in aspects:
                aspect_sent = AspectBasedSent(sent, [asp])
                sents_aspects.append(aspect_sent)
        else:
            aspect_sent = AspectBasedSent(sent, aspects)
            sents_aspects.append(aspect_sent)
    return sents_aspects
