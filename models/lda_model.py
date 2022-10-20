import pandas as pd

from models.model import Model
from utils.data_structures import OutputData

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

class LDAModel(Model):
    """
    Brzydka i niekompletna wstępna implemetacja przykładowego modelu.
    """

    def __init__(self, parameters=None):
        super().__init__(parameters)
        if parameters is None:
            self.init_default_parameters()


    def fit(self, data):
        super().fit(data)
        self.dictionary = Dictionary(data.texts)
        self.dictionary.filter_extremes(**self.parameters["filter_extremes"])
        corpus = [self.dictionary.doc2bow(text) for text in data.texts]
        lda = LdaModel(corpus, **self.parameters["lda"])
        self.model = lda

    def get_topics(self):
        topics = self.model.show_topics()

        output = OutputData()
        for i in range(0, self.model.num_topics):
           output.add_topic([(self.dictionary[token], score) for (token, score) in self.model.get_topic_terms(i, topn=10)])

        return output
    
    def init_default_parameters(self):
        self.parameters = { "filter_extremes" : {"no_below" : 20,
                                                 "no_above" : 0.4},
                            "lda": {"num_topics" : 10,
                                     "alpha" : "auto",
                                     "eval_every" : 2}}

