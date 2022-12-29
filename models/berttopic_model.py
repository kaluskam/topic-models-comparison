from msilib import init_database

import numpy as np
import pandas as pd
import hdbscan
from .model import Model
from utils.data_structures import OutputData

from bertopic import BERTopic


class BERTopicModel(Model):
    """
    Implementation of a BERTopic model. Extends abstract Model class.
    """
    def __init__(self, parameters=None):
        super().__init__(parameters)
        if parameters is None:
            self.init_default_parameters()

    def fit(self, data, n_topics = None):
        n_topics = n_topics if n_topics is not None else 10
        super().fit(data)
        self.data = data
        bert = BERTopic(**self.parameters["bertopic"])
        self.topic_ids, _ = bert.fit_transform([str(text) for text in data.texts])
        bert.reduce_topics([str(text) for text in data.texts], nr_topics=n_topics)
        self.model = bert

    def get_output(self):
        self.output = OutputData(self.data)
        for i in range(0, len(self.model.get_topics()) - 1):
            words = [word for (word, score) in self.model.get_topic(i)]
            word_scores = [score for (word, score) in self.model.get_topic(i)]
            frequency = self.model.get_topic_freq(i)
            self.output.add_topic(words, word_scores, frequency)

        self._match_texts_with_topics()
        self.output.topic_word_matrix = self.output.create_topic_word_matrix()
        return self.output

    def _match_texts_with_topics(self):
        self.topic_ids = np.array(self.topic_ids) + 1
        self.output.add_texts_topics(np.arange(1, len(self.topic_ids) + 1), self.topic_ids)

    def init_default_parameters(self):
        clusterer = hdbscan.HDBSCAN(algorithm='best', approx_min_span_tree=True,
                                    gen_min_span_tree=False, leaf_size=30, metric='euclidean', min_cluster_size=25,
                                    min_samples=20, cluster_selection_epsilon=0.5, p=None)
        self.parameters = {"bertopic": {"n_gram_range": (1, 1), "hdbscan_model": clusterer, "nr_topics": "auto"}}

    def save(self, filepath):
        super().save(filepath)
