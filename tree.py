from aspect import Aspect

op_tags = ["JJ", "JJR", "JJS", "VB", "VBD", "VBN", "VBG", "VBP", "VBZ"]
noun_tags = ["NN", "NNS", "NNP", "NNPS"]


class Tree(object):
    """
    # Tree class to represent constituent trees.
    """

    def __init__(self, label, tag=None):
        """
        Constructor for the tree object.
        :param label: The sentiment label of this node (if exists).
        :param tag: A constitue syntax tag of this node.
        """
        self._children = []
        self._label = label
        self._tag = tag

        self._papa = None
        self._height = None
        self._phrase = None
        self._phraselen = None
        self._has_op = False
        self._asp_num = 0

    @property
    def tag(self):
        """
        Get function for the syntax tag of this node.
        :return: the syntax tag of this node.
        """
        return self._tag

    @tag.setter
    def tag(self, value):
        """
        Set function of the syntax tag of this node.
        :param value: The syntax tag of this node.
        """
        self._tag = value

    @property
    def phraselen(self):
        """
        Get function of the phrase length of this node.
        :return: the phrase length of this node.
        """
        return self._phraselen

    @property
    def phrase(self):
        """
        Get function the phrase of this node.
        :return: the phrase length of this node.
        """
        return self._phrase

    @phrase.setter
    def phrase(self, value):
        """
        Set function of the phrase of this node.
        :param value: the phrase of this node.
        """
        self._phraselen = len(value.split(' '))
        self._phrase = value

    @property
    def children(self):
        """
        Get function of the number of children of this node.
        :return: the number of children of this node.
        """
        return len(self._children)

    def add_child(self, child):
        """
        A function that adds a new child to this node.
        :param child: the child to add to the tree.
        """
        self._children.append(child)

    def get_child(self, idx):
        """
        A function the returns the child in index idx of the root of the tree.
        :param idx: the index of the child of the root of the tree.
        :return: the child of this node according to the index given.
        """
        return self._children[idx]

    @property
    def label(self):
        """
        Get function for the label of this node.
        :return: the label of this node.
        """
        return self._label

    @label.setter
    def label(self, value):
        """
        Set function for the label of this node.
        :param value: the label to be set.
        """
        self._label = value

    @property
    def papa(self):
        """
        Get function for the parent of the node.
        :return: the parent of the node.
        """
        return self._papa

    @papa.setter
    def papa(self, value):
        """
        Set function for the parent of the node.
        :param value: The parent to be set.
        """
        self._papa = value

    @property
    def height(self):
        """
        Get function of the height of this node
        :return: The height of this node.
        """
        return self._height

    @height.setter
    def height(self, value):
        """
        Set function of the height of this node.
        :param value: The height of this node to be set.
        """
        self._height = value

    @staticmethod
    def is_leaf():
        """
        Function that returns if the this tree is a leaf.
        :return: Always false since this class is not a leaf.
        """
        return False

    @property
    def has_op(self):
        """
        Get function that returns true if this node governs an opinion, else false
        :return: A boolean to indicate if this node governs an opinion.
        """
        return self._has_op

    @has_op.setter
    def has_op(self, value):
        """
        Set function to set the boolean that indicates if this node governs an opinion.
        :param value: A boolean to indicate if this node governs an opinion.
        """
        self._has_op = value

    @property
    def asp_num(self):
        """
        Get function that returns the number of aspects governed by this node.
        :return: The number of aspects governed by this node.
        """
        return self._asp_num

    def add_asp(self):
        """
        Add one to the number of aspects governed by this node.
        """
        self._asp_num += 1


class Leaf(object):
    """
    The leaf class represents a leaf of a tree.
    """

    def __init__(self, label, word, tag=None):
        """
        Constructor.
        :param label: The sentiment label of this node (if exists).
        :param word: The word represented in this leaf.
        :param tag: A constitute syntax tag of this node.
        """
        self._label = label
        self._word = word
        self._papa = None
        self._tag = tag

    @property
    def tag(self):
        """
        Get function for the pos tag of this leaf.
        :return: The pos tag of this leaf
        """
        return self._tag

    @tag.setter
    def tag(self, value):
        """
        Set function for the pos tag of this leaf
        :param value: The pos tag of this leaf
        """
        self._tag = value

    @property
    def height(self):
        """
        Get function for the height of this node.
        :return: 0, since this is a leaf node.
        """
        return 0

    @property
    def label(self):
        """
        Get function for the sentiment label of the leaf.
        :return: The sentiment label of the leaf.
        """
        return self._label

    @label.setter
    def label(self, value):
        """
        Set function for the sentiment label of the leaf.
        :param value: The label to be set.
        """
        self._label = value

    @property
    def papa(self):
        """
        Get function for the parent of the leaf.
        :return: The parent of the leaf.
        """
        return self._papa

    @papa.setter
    def papa(self, value):
        """
        Set function for the parent of the leaf.
        :param value: the parent to be set.
        """
        self._papa = value

    @property
    def word(self):
        """
        Get function for the word of the leaf.
        :return: the word of the leaf.
        """
        return self._word

    @staticmethod
    def is_leaf():
        """
        Function to check if the this tree is a leaf.
        :return: Always true since this class is a leaf.
        """
        return True

    @property
    def has_op(self):
        """
        Get function that returns a boolean that indicates if the leaf governs an opinion.
        :return: Always false since this is a leaf and does not have decedents, and can't govern
        an opinion.
        """

        return False

    @property
    def asp_num(self):
        """
        Get function for the number of aspects linked to this node.
        :return: Zero.
        """
        return 0

    @property
    def phraselen(self):
        """
        Get function for the phrase length of this node.
        :return: 1, since this is a leaf node.
        """
        return 1

    @property
    def phrase(self):
        """
        Get the phrase of this node.
        :return: the phrase of this node
        """
        return self._word

    @phrase.setter
    def phrase(self, value):
        """
        Set the phrase of this node.
        :param value: the phrase of this node
        """
        self._word = value


def count_node_aspects(tree, aspects):
    """
    A recursive function that counts the number of aspects governed by each node.
    :param tree: A node.
    :param aspects: The aspects in a sentence.
    """
    if not tree.is_leaf():
        for asp in aspects:
            if asp.expression_term in tree.phrase:
                tree.add_asp()
        for i in range(tree.children):
            count_node_aspects(tree.get_child(i), aspects)


def find_pot_aspects(tree, pot_aspects, tags):
    """
    A recursive function that finds each noun in a sentence and appends them to a list.
    :param tree: A node.
    :param pot_aspects: A list of the current noun phrases found in a tree.
    :param tags: The tags of nouns.
    """
    if tree.is_leaf():
        if tree.tag in tags:
            pot_aspects.append(Aspect(None, tree.phrase, 6))
    else:
        for i in range(tree.children):
            child = tree.get_child(i)
            find_pot_aspects(child, pot_aspects, tags)


def fill_govern_tag(tree, tags, aspects=list()):
    """
    A recursive function that sets if a node governs an opinion or not on each node of a tree.
    :param tree: A node.
    :param tags: A list of tags that might indicate an opinion (verbs or adjectives)
    :param aspects: A list of aspects.
    :return: Boolean to indicate if the current node governs an opinion or not.
    """
    if tree.is_leaf():
        if tree.tag in tags:
            for asp in aspects:
                if tree.phrase in asp.expression_term or asp.expression_term in tree.phrase:
                    return False
            return True
    else:
        for i in range(tree.children):
            child = tree.get_child(i)
            if fill_govern_tag(child, tags, aspects):
                tree.has_op = True
        if tree.has_op:
            return True
    return tree.has_op


def decompse_sent(tree, aspects):
    """
    A recursive function to decompose a sentence according to a few heuristics.
    :param tree: A constituent tree of a sentence.
    :param aspects: A list of aspects.
    """
    if len(aspects) == 1:
        if tree.papa is None:
            aspects[0].pred_sentiment = tree.label
            aspects[0].linked_node = tree
        else:
            if tree.has_op:
                if aspects[0].pred_sentiment is None:
                    aspects[0].pred_sentiment = tree.label
                    aspects[0].shared = tree.asp_num
                    aspects[0].linked_node = tree

    if len(aspects) > 1:
        if not tree.is_leaf():
            for i in range(tree.children):
                child = tree.get_child(i)
                child_asps = list()
                for asp in aspects:
                    if child.is_leaf():
                        if asp.expression_term == child.phrase:
                            child_asps.append(asp)
                    else:
                        if asp.expression_term in child.phrase:
                            child_asps.append(asp)
                decompse_sent(child, child_asps)
        if tree.has_op or tree.papa is None:
            for asp in aspects:
                if asp.pred_sentiment is None or asp.linked_node.asp_num == tree.asp_num:
                    asp.pred_sentiment = tree.label
                    asp.linked_node = tree


