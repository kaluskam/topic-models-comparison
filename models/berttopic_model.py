from msilib import init_database
import pandas as pd

from .model import Model
from utils.data_structures import OutputData

from bertopic import BERTopic

class BERTopicModel(Model):
    """
    Brzydka i niekompletna wstępna implemetacja przykładowego modelu.
    """

    def __init__(self, parameters = None):
        super().__init__(parameters)
        if parameters is None:
            self.init_default_parameters()

    def fit(self, data):
        super().fit(data)
        self.data = data
        bert = BERTopic(**self.parameters["bertopic"])
        bert.fit_transform([" ".join(doc) for doc in data.texts])
        self.model = bert

    def get_topics(self):
        output = OutputData(self.data)
        for i in range(0, len(self.model.get_topics())-1):
            words = [word for (word, score) in self.model.get_topic(i)]
            word_scores = [score for (word, score) in self.model.get_topic(i)]
            frequency = self.model.get_topic_freq(i)
            output.add_topic(words, word_scores, frequency)
        output.topic_word_matrix = output.create_topic_word_matrix()
        return output
    
    def init_default_parameters(self):
        self.parameters = {"bertopic": {"n_gram_range": (1,1)}}