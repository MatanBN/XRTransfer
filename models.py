import pickle

from import_dy import dy


class BaseModel(object):
    """
    BaseModel class is an abstract class that every model should inherit from, it contains basic
    functions which would be necessary for every model.
    """

    def save_model(self, model_file):
        """
        Save the model.
        :param model_file: The model file name to soad.
        """
        self._model.save(model_file)

    def load_model(self, model_file):
        """
        Load the model.
        :param model_file: The model file name to load.
        """
        self._model.populate(model_file)

    def load_meta(self, meta_file):
        """
        Load the w2i dictionary.
        :param meta_file: The dictionary file.
        """
        self._w2i = pickle.load(open(meta_file, "rb"))

    def save_meta(self, meta_file):
        """
        Save the w2i dictionary.
        :param meta_file: The file of the dicitonary.
        """
        pickle.dump(self._w2i, open(meta_file, "wb"))

    def get_model(self):
        """
        Get the parameter collection of a model.
        :return: The parameter collection of a model.
        """
        return self._model


class LinearClassifier(BaseModel):
    """
    Linear classifier class is a linear classifier used on top of some deep model (mlp with one layer).
    """
    def __init__(self, input_dim, output_dim, model):
        """
        Constructor.
        :param input_dim: The input dimension for the linear classifier.
        :param output_dim: The number of labels.
        :param model: The parameter collection of the model it used on top of.
        """
        self._model = model
        self._W = model.add_parameters((output_dim, input_dim))
        self._b = model.add_parameters(output_dim)

    def classify(self, h, train=False):
        """
        Classify a vector to one of the labels.
        :param h: A vector to classify.
        :param train: Boolean to indicate if this is train mode or not.
        :return: The probabilities given by the classifier.
        """
        if train:
            h = dy.dropout(h, 0.5)
        prediction = dy.softmax(self._W * h + self._b)
        return prediction


class BiLSTM(BaseModel):
    """
    BiLSTM class, an implementation of the BiLSTM network, contains model parameters and functions
    for the forward pass.
    """

    def __init__(self, layers, in_dim, lstm_dim, word_vocab, model, pre_trained=None):
        """
        Constructor for the BiLSTMNetwork
        :param layers: The number of layers of the BiLSTM (default is one).
        :param in_dim: The input dim of the model (the dimension of the embedding vectors).
        :param lstm_dim: Number of the dimension of the output of the lstm.
        :param word_vocab: The word 2 key dictionary.
        :param model: The parameter collection.
        :param pre_trained: Pretrained embedding vectors.
        """
        self._model = model
        if pre_trained is None:
            self._E = model.add_lookup_parameters((len(word_vocab), in_dim))
        else:
            self._E = model.lookup_parameters_from_numpy(pre_trained)
        self._fwd_RNN_first = dy.VanillaLSTMBuilder(layers, in_dim, lstm_dim, model)
        self._bwd_RNN_first = dy.VanillaLSTMBuilder(layers, in_dim, lstm_dim, model)
        self._classifier = LinearClassifier(2 * lstm_dim, 3, model)
        self._w2i = word_vocab

    def __call__(self, sequence):
        """
        Standard classification output, i.e. read from start to end and from end to start and
        concatenate the two vectors.
        :param sequence: The sequence to read.
        :return: A concatenation of the forward and backward pass.
        """
        fwd_states, bwd_states = self.encode_fwd_bwd(sequence)
        return dy.concatenate([fwd_states[-1], bwd_states[-1]])

    def embed(self, sequence):
        """
        Embed the input words to word vectors.
        :param sequence: A sequence of words.
        :return: A sequence of embedding vectors.
        """
        words = sequence.split(' ')
        vecs = [self._E[self._w2i[i]] if i in self._w2i else self._E[self._w2i["UNK"]]
                for i in words]
        return vecs

    def encode_vecs(self, vecs, lstm):
        """
        Encode a sequence of vectors using the lstm equations.
        :param vecs: A sequence of vectors.
        :param lstm: An lstm to encode the sequence vectors.
        :return: The states of the lstm.
        """
        initial_state = lstm.initial_state()
        states = initial_state.transduce(vecs)
        return states

    def encode_fwd_bwd(self, sequence):
        """
        Encode a sequence of words from start to end and from end to start.
        :param sequence: a sequence of words.
        :return: A list of states start to end and a list of states from end to start.
        """
        vecs = self.embed(sequence)
        fwd_states = self.encode_vecs(vecs, self._fwd_RNN_first)
        bwd_states = self.encode_vecs(vecs[::-1], self._bwd_RNN_first)
        return fwd_states, bwd_states

    def encode(self, sequence):
        """
        Encode a sequence of words from start to end and end to start and concatenate the states
        of the forward and backward passes.
        :param sequence: A sequence of words.
        :return: A concatenation of the forward and backward states.
        """
        fwd_states, bwd_states = self.encode_fwd_bwd(sequence)
        bwd_states = bwd_states[::-1]
        return [dy.concatenate([fwd_states[i], bwd_states[i]]) for i in range(len(fwd_states))]

    def encode_aspects(self, sample):
        """
        Encode a list of aspects using the expression phrase that is linked to them.
        :param sample: a sample of sentence and its aspects.
        :return: A list of outputs corresponding to the aspects.
        """
        outputs = list()
        for asp in sample.aspects:
            output = self(asp.expression_phrase)
            outputs.append(output)
        return outputs

    def classify(self, expr, train=False):
        """
        Classify a vector using the linear classifier.
        :param expr: The vector to classify.
        :param train: Boolean to indicate if this is during training or not.
        :return: The probabilities given by the classifier.
        """
        return self._classifier.classify(expr, train=train)

    @property
    def w2i(self):
        """
        Get function for the word to dictionary (w2i) attribute.
        :return: The w2i attribute
        """
        return self._w2i

    @property
    def E(self):
        """
        Get function for the embedding matrix.
        :return: The embedding matrix.
        """
        return self._E


class LSTMAtt(BaseModel):
    """
    LSTM Att class is an attention based bilstm model. The model adds attention parameters on top
    of a bilstm and uses the word embeddings of an aspect to calculate an attention score
    on the bilstm states.
    """

    def __init__(self, seq_model, model, in_dim, lstm_dim):
        """
        Constructor.
        :param seq_model: The BiLSTM model the attention is based on.
        :param model: Parameter collection.
        :param in_dim: The input dimension of the BiLSTM (dimension of word embeddings).
        :param lstm_dim: The output dimension the BiLSTM.
        """
        self._model = model
        self._seq_model = seq_model
        self._w2i = seq_model.w2i
        self._E = seq_model.E
        self._Wa = model.add_parameters((2 * lstm_dim, in_dim))
        self._Wd = model.add_parameters((2 * lstm_dim, 4 * lstm_dim))
        self._bd = model.add_parameters(2 * lstm_dim)

    def __call__(self, sequence, aspect):
        """
        Encode an aspect representation based on the sentence and the aspect.
        :param sequence: A sequence of words.
        :param aspect: The aspect to encode.
        :return: The representation of the aspect.
        """
        sequence_vec = self._seq_model.encode(sequence)
        asp_words = aspect.expression_term.split(' ')
        asp_vecs = [self._E[self._w2i[word]] if word in self._w2i else self._E[self._w2i["UNK"]]
                    for word in asp_words]
        t = dy.average(asp_vecs)

        seq_conc = dy.concatenate_cols(sequence_vec)
        beta = dy.tanh(dy.transpose(seq_conc) * self._Wa * t)
        alpha = dy.softmax(beta)
        z_list = list()
        for i in range(len(sequence_vec)):
            z_list.append(sequence_vec[i] * alpha[i])
        z = dy.esum(z_list)
        return z

    def encode_aspects(self, sample):
        """
        Encode the aspects in a sentence.
        :param sample: A sample containing a sentence and its aspects.
        :return: the representation of each aspect in the sample.
        """
        outputs = list()
        for asp in sample.aspects:
            output = self(sample.sent[0], asp)
            outputs.append(output)
        return outputs

    def classify(self, expr, train=False):
        """
        Classify an aspect expression using a linear classifier.
        :param expr: The expression vector of an aspect.
        :param train: Boolean to indicate if this is in training or not.
        :return: The probabilities of the classifier.
        """
        return self._seq_model._classifier.classify(expr, train=train)
