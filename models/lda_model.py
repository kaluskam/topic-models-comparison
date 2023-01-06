import pandas as pd
import numpy as np

from models.model import Model
from utils.data_structures import OutputData

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel


class LDAModel(Model):
    """
    Implementation of a Latent Dirichlet Allocation model from gensim library. Extends abstract Model class.
    """

    def __init__(self, parameters=None):
        super().__init__(parameters)
        if parameters is None:
            self.init_default_parameters()
        self.output = None

    def fit(self, data, n_topics = 10):
        super().fit(data, n_topics)
        self.parameters["lda"]["num_topics"] = n_topics
        self.data = data
        self.dictionary = Dictionary(data.texts)
        self.dictionary.filter_extremes(**self.parameters["filter_extremes"])
        self.corpus = [self.dictionary.doc2bow(text) for text in data.texts]
        self.model = LdaModel(self.corpus, **self.parameters["lda"])

    def get_output(self):
        self.output = OutputData(self.data)

        topic_distribution = self.model.get_topics()
        frequencies = np.sum(topic_distribution, axis=0)
        frequencies /= np.sum(frequencies)

        for i in range(0, self.model.num_topics):
            words = [self.dictionary[token] for (token, score) in
                     self.model.get_topic_terms(i, topn=10)]
            word_scores = [score for (token, score) in
                           self.model.get_topic_terms(i, topn=10)]
            frequency = frequencies[i]
            self.output.add_topic(words, word_scores, frequency)

        self._match_texts_with_topics()
        self.output.topic_word_matrix = self.output.create_topic_word_matrix()

        return self.output

    def _match_texts_with_topics(self):
        topic_ids = []
        text_ids = np.arange(1, len(self.corpus))
        for i in text_ids:
            topic_ids.append(np.argmax(
                np.array(self.model.get_document_topics(self.corpus[i]))[:, 1]))
        self.output.add_texts_topics(text_ids, topic_ids)

    def init_default_parameters(self):
        self.parameters = {"filter_extremes": {"no_below": 2, #20
                                               "no_above": 0.1}, #0.4
                           "lda": {"num_topics": 10,
                                   "alpha": "auto",
                                   "eval_every": 2,
                                   "passes": 70}}
    def save(self, filepath):
        super().save(filepath)