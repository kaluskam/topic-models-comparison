from msilib import init_database

import numpy as np
import pandas as pd

from .model import Model
from utils.data_structures import OutputData

from bertopic import BERTopic


class BERTopicModel(Model):

    def __init__(self, parameters=None):
        super().__init__(parameters)
        if parameters is None:
            self.init_default_parameters()

    def fit(self, data):
        super().fit(data)
        self.data = data
        bert = BERTopic(**self.parameters["bertopic"])
        self.topic_ids, _ = bert.fit_transform([" ".join(doc) for doc in data.texts])

        self.model = bert

    def get_output(self):
        self.output = OutputData(self.data)
        for i in range(0, len(self.model.get_topics()) - 1):
            words = [word for (word, score) in self.model.get_topic(i)]
            word_scores = [score for (word, score) in self.model.get_topic(i)]
            frequency = self.model.get_topic_freq(i)
            self.output.add_topic(words, word_scores, frequency)

        #self._match_texts_with_topics()
        return self.output

    def _match_texts_with_topics(self):
        self.topic_ids = np.array(self.topic_ids) + 1
        self.output.add_texts_topics(np.arange(1, len(self.topic_ids) + 1), self.topic_ids)

    def init_default_parameters(self):
        self.parameters = {"bertopic": {"n_gram_range": (1, 1)}}
