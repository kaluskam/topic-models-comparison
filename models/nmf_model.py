import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .model import Model
from utils.data_structures import OutputData


class NMFModel(Model):

    def __init__(self, parameters=None):
        super().__init__(parameters)
        if parameters is None:
            self.init_default_parameters()
        self.tfidf_vectorizer = TfidfVectorizer(**self.parameters['tfidf'])
        self.output = None

    def fit(self, data, n_topics=10):
        super().fit(data, n_topics)
        self.data = data
        tfidf = self.tfidf_vectorizer.fit_transform(data.texts)
        self.parameters['nmf']['n_components'] = n_topics
        self.model = NMF(**self.parameters['nmf'])
        self.W = self.model.fit_transform(tfidf)
        self.H = self.model.components_

    def get_output(self):
        components_df = pd.DataFrame(self.model.components_,
                                     columns=self.tfidf_vectorizer.get_feature_names()) #na nowej wersji jest out
        self.output = OutputData(self.data)

        frequencies = np.sum(self.W, axis=1)
        frequencies /= np.sum(frequencies)

        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            words = [ind for ind in tmp.nlargest(10).index]
            word_scores = [tmp[ind] for ind in tmp.nlargest(10).index]
            self.output.add_topic(words, word_scores, frequencies[topic])

        self._match_texts_with_topics()
        #self.output.topic_word_matrix = self.output.create_topic_word_matrix()

        return self.output

    def _match_texts_with_topics(self):
        text_ids = np.arange(1, len(self.data.texts) + 1)
        topic_ids = np.argmax(self.W, axis=1)
        self.output.add_texts_topics(text_ids, topic_ids)

    def choose_number_of_topics(self):
        pass

    def init_default_parameters(self):
        self.parameters = {'tfidf': {'preprocessor': ' '.join},
                           'nmf': {'n_components': 5}}
    def save(self, filepath):
        super().save(filepath)
