class AspectBasedSent(object):
    """
    This class is a container class for a sentence and its aspects.
    """
    def __init__(self, sent, aspects):
        """
        Constructor for the AspectBasedSent object.
        :param sent: The sentence of a sample.
        :param aspects: The aspects contained within a sentence.
        """
        self.sent = sent
        self.aspects = aspects


class Aspect(object):
    """
    This class is responsible for containing the attributes of a given aspect.
    The aspect object contains the following attributes:
    gold_sentiment: the gold sentiment of the aspect
    pred_sentiment: a prediction sentiment given to it by a classifier
    expression_phrase: The expression phrase that represents the aspect (This is usually
        noisy a bit, since it is retrieved with parsing and a few heuristics).
    shared: the number of aspects that are contained in the same expression term it was found in.
    linked_node: The tree node it was linked to.
    expression_phrase: The expression phrase that represents the aspect.
    conflicted: Boolean to indicate if this aspect is conflicted (i.e. there is some other aspect
    in the same sentence with the same term which contains a different sentiment, similar to previous
    work, we remove such aspects).
    """

    def __init__(self, expression_phrase, expression_term, gold_sentiment):
        """
        Constructor for the aspect class.
        :param expression_phrase: The expression phrase that represents the aspect.
        :param expression_term: The linguistic term that expressed the aspect (this does not always exist).
        :param gold_sentiment: The gold sentiment of the aspect.
        """
        self._gold_sentiment = gold_sentiment
        self._pred_sentiment = None
        self._shared = None
        self._expression_term = expression_term
        self._linked_node = None
        self._expression_phrase = expression_phrase
        self.conflicted = True

    @property
    def gold_sentiment(self):
        """
        Get function for the gold_sentiment attribute.
        :return: The gold_sentiment attribute.
        """
        return self._gold_sentiment

    @property
    def pred_sentiment(self):
        """
        Get function for the pred_sentiment attribute.
        :return: The pred_sentiment attribute.
        """
        return self._pred_sentiment

    @pred_sentiment.setter
    def pred_sentiment(self, value):
        """
        Set function for the pred_sentiment attribute.
        :param value: The value to set pred_sentiment with.
        """
        self._pred_sentiment = value

    @property
    def shared(self):
        """
        Get function for the shared attribute.
        :return: The shared attribute.
        """
        return self._shared

    @shared.setter
    def shared(self, value):
        """
        Set function for the shared attribute.
        :param value: The value to set shared with.
        """
        self._shared = value

    @property
    def expression_term(self):
        """
        Get function for the expression_term attribute.
        :return: The expression_term attribute.
        """
        return self._expression_term

    @property
    def linked_node(self):
        """
        Get function for the linked_node attribute.
        :return: The linked_node attribute.
        """
        return self._linked_node

    @linked_node.setter
    def linked_node(self, value):
        """
        Set function for the linked_node attribute.
        :param value: The value to set linked_node with.
        """
        self._linked_node = value

    @property
    def expression_phrase(self):
        """
        Get function for the expression_phrase attribute.
        :return: The expression_phrase attribute.
        """
        return self._expression_phrase

    @expression_phrase.setter
    def expression_phrase(self, value):
        """
        Set function for the expression_phrase attribute.
        :param value: The value to set expression_phrase with.
        """
        self._expression_phrase = value



