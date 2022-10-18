import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from model import Model
from utils.data_structures import OutputData


class NMFModel(Model):
    """
    Brzydka i niekompletna wstępna implemetacja przykładowego modelu.
    """

    def __init__(self, parameters=None):
        super().__init__(parameters)
        if parameters is None:
            self.init_default_parameters()
        self.tfidf_vectorizer = TfidfVectorizer(**self.parameters['tfidf'])

    def fit(self, data):
        super().fit(data)
        tfidf = self.tfidf_vectorizer.fit_transform(data.texts)
        self.model = NMF(**self.parameters['nmf'])
        self.model.fit_transform(tfidf)

    def get_topics(self):
        components_df = pd.DataFrame(self.model.components_,
                                     columns=self.tfidf_vectorizer.get_feature_names_out())
        output = OutputData()
        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            top = [(ind, tmp[ind]) for ind in tmp.nlargest(10).index]
            output.add_topic(top)

        return output

    def choose_number_of_topics(self):
        pass

    def init_default_parameters(self):
        self.parameters = {'tfidf': {'preprocessor': ' '.join},
                           'nmf': {'n_components': 10}}
