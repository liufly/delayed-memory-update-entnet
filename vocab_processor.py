import numpy as np
import tensorflow as tf
import collections

class LabelVocabulary(tf.contrib.learn.preprocessing.CategoricalVocabulary):
    def __init__(self, support_reverse=True):
        self._mapping = {}
        self._support_reverse = support_reverse
        if support_reverse:
            self._reverse_mapping = []
        self._freq = collections.defaultdict(int)
        self._freeze = False

    def get(self, category):
        """Returns word's id in the vocabulary.
        If category is new, creates a new id for it.
        Args:
        category: string or integer to lookup in vocabulary.
        Returns:
        interger, id in the vocabulary.
        """
        if category not in self._mapping:
            if self._freeze:
                assert False # should not happen
                # return self._mapping[self._unknown_token]
            self._mapping[category] = len(self._mapping)
            if self._support_reverse:
                self._reverse_mapping.append(category)
        return self._mapping[category]

    def trim(self, min_frequency, max_frequency=-1):
        """Trims vocabulary for minimum frequency.
        Remaps ids from 1..n in sort frequency order.
        where n - number of elements left.
        Args:
        min_frequency: minimum frequency to keep.
        max_frequency: optional, maximum frequency to keep.
            Useful to remove very frequent categories (like stop words).
        """
        # no need to trim for label vocab
        return

class EmbeddingVocabulary(tf.contrib.learn.preprocessing.CategoricalVocabulary):
    def __init__(self,
                 in_file,
                 binary=False,
                 padding_token="<PAD>",
                 unknown_token="<UNK>",
                 support_reverse=True):
        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._mapping = {padding_token: 0, unknown_token: 1}
        self._support_reverse = support_reverse
        if support_reverse:
            self._reverse_mapping = [padding_token, unknown_token]
        # no need to count frequency
        # self._freq = collections.defaultdict(int)
        self._load_embeddings(in_file, binary=binary)
        # freeze the vocabulary once the embeddings have been loaded
        self._freeze = True

    def _load_embeddings(self, in_file, binary=False):
        # emb = word2vec.Word2Vec.load_word2vec_format(in_file, binary=binary)
        with open(in_file) as in_f:
            nb_words, nb_dim = None, None
            for line in in_f:
                line = line.strip()
                attrs = line.split(' ')
                if len(attrs) == 2:
                    nb_words = int(attrs[0])
                    nb_dim = int(attrs[1])
                    self._embeddings = np.zeros((nb_words + 2, nb_dim), dtype=np.float32)
                    continue
                word = attrs[0]
                emb = map(float, attrs[1:])
                self._mapping[word] = len(self._mapping) if not self._support_reverse else len(self._reverse_mapping)
                self._embeddings[self._mapping[word], :] = emb
                if self._support_reverse:
                    self._reverse_mapping.append(word)

            unk = np.mean(self._embeddings[2:], axis=0)
            self._embeddings[self._mapping[self._unknown_token]] = unk

    def _get_mean_embeddings(self, emb):
        syn0 = emb.syn0
        return np.mean(syn0, axis=0)

    @property
    def embeddings(self):
        return self._embeddings

    def freeze(self, freeze=True):
        """Freezes the vocabulary, after which new words return unknown token id.
        Args:
        freeze: True to freeze, False to unfreeze.
        """
        self._freeze = True # should always be True after __init__

    def get(self, category):
        """Returns word's id in the vocabulary.
        If category is new, creates a new id for it.
        Args:
            category: string or integer to lookup in vocabulary.
        Returns:
            interger, id in the vocabulary.
        """
        if category not in self._mapping:
            if self._freeze:
                return self._mapping[self._unknown_token]
            assert False # should not happey
            self._mapping[category] = len(self._mapping)
            if self._support_reverse:
                self._reverse_mapping.append(category)
        return self._mapping[category]

    def add(self, category, count=1):
        """Adds count of the category to the frequency table.
        Args:
        category: string or integer, category to add frequency to.
        count: optional integer, how many to add.
        """
        # do nothing
        return

    def trim(self, min_frequency, max_frequency=-1):
        """Trims vocabulary for minimum frequency.
        Remaps ids from 1..n in sort frequency order.
        where n - number of elements left.
        Args:
            min_frequency: minimum frequency to keep.
            max_frequency: optional, maximum frequency to keep.
                Useful to remove very frequent categories (like stop words).
        """
        # don't trim embedding vocab
        return

class EmbeddingVocabularyProcessor(tf.contrib.learn.preprocessing.VocabularyProcessor):

    def __init__(self,
                 max_document_length,
                 vocabulary,
                 min_frequency=0,
                 tokenizer_fn=None):
        self.max_document_length = max_document_length
        self.vocabulary_ = vocabulary # EmbeddingVocabulary object
        self.min_frequency = min_frequency

    @staticmethod
    def tokenize(sentence):
        # for value in iterator:
        #     yield value.split(' ')
        return sentence.split(' ')

    def fit(self, sentences, unused_y=None):
        # do nothing given that the embeddings have already been
        # initialized in EmbeddingVocabulary
        for sentence in sentences:
            for token in sentence:
                self.vocabulary_.add(token)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        self.vocabulary_.freeze()
        return self

    def transform(self, sentences):
        '''
        Args:
            sentences: list of list of words
        Returns:
            indices: list of list of word indices
        '''
        word_ids = np.zeros((len(sentences), self.max_document_length), np.int32)
        for i, sentence in enumerate(sentences):
            # word_ids = np.zeros(self.max_document_length, np.int32)
            for j, token in enumerate(sentence):
                if j >= self.max_document_length:
                    break
                word_ids[i, j] = self.vocabulary_.get(token)
        return word_ids

    def reverse(self, sentences):
        """Reverses output of vocabulary mapping to words.
        Args:
            sentences: list of list of word indices
        Returns:
            output: list of list of words
        """
        output = []
        for sentence in sentences:
            output.append(
                [self.vocabulary_.reverse(word_id) for word_id in sentence]
            )
        return output

class LabelVocabularyProcessor(tf.contrib.learn.preprocessing.VocabularyProcessor):

    def __init__(self,
                #  max_document_length,
                 vocabulary,
                 min_frequency=0,
                 tokenizer_fn=None):
        self.vocabulary_ = vocabulary # EmbeddingVocabulary object
        self.min_frequency = min_frequency

    @staticmethod
    def tokenize(sentence):
        # for value in iterator:
        #     yield value.split(' ')
        return sentence.split(' ')

    def fit(self, sentences, unused_y=None):
        # do nothing given that the embeddings have already been
        # initialized in EmbeddingVocabulary
        for label in sentences:
            self.vocabulary_.add(token)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        self.vocabulary_.freeze()
        return self

    def transform(self, sentences):
        '''
        Args:
            sentences: list of list of words
        Returns:
            indices: list of list of word indices
        '''
        label_ids = np.full((len(sentences)), -1, dtype=np.int32)
        for i, label in enumerate(sentences):
            label_ids[i] = self.vocabulary_.get(label)
            # for j, token in enumerate(sentence):
            #     if j >= self.max_document_length:
            #         break
            #     label_ids[i, j] = self.vocabulary_.get(token)
        return label_ids

    def reverse(self, sentences):
        """Reverses output of vocabulary mapping to words.
        Args:
            sentences: list of list of word indices
        Returns:
            output: list of list of words
        """
        output = []
        for label_id in sentences:
            output.append(
                self.vocabulary_.reverse(label_id)
            )
        return output
