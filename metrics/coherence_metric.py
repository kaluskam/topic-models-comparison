from utils.data_structures import *
from metrics.metric import Metric

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


class UMassCoherenceMetric(Metric):

    def __init__(self, flag=False, range=(-14, 14), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        topics = outputData.get_topics()
        dictionary = Dictionary(inputData.texts)
        corpus = [dictionary.doc2bow(doc) for doc in inputData.texts]

        cm = CoherenceModel(topics=topics,
                            texts=inputData.texts,
                            dictionary=dictionary,
                            corpus=corpus,
                            coherence="u_mass",
                            **self.parameters["coherencemodel"])
        return cm.get_coherence()

    def init_default_parameters(self):
        self.parameters = {"coherencemodel": {}}


class CVCoherenceMetric(Metric):

    def __init__(self, flag=True, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        topics = outputData.get_topics()
        dictionary = Dictionary(inputData.texts)
        corpus = [dictionary.doc2bow(doc) for doc in inputData.texts]

        cm = CoherenceModel(topics=topics,
                            texts=inputData.texts,
                            dictionary=dictionary,
                            corpus=corpus,
                            coherence="c_v",
                            **self.parameters["coherencemodel"])
        return cm.get_coherence()

    def init_default_parameters(self):
        self.parameters = {"coherencemodel": {}}


class CUCICoherenceMetric(Metric):

    def __init__(self, flag=True, range=(-1, 1), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        topics = outputData.get_topics()
        dictionary = Dictionary(inputData.texts)
        corpus = [dictionary.doc2bow(doc) for doc in inputData.texts]

        cm = CoherenceModel(topics=topics,
                            texts=inputData.texts,
                            dictionary=dictionary,
                            corpus=corpus,
                            coherence="c_uci",
                            **self.parameters["coherencemodel"])
        return cm.get_coherence()

    def init_default_parameters(self):
        self.parameters = {"coherencemodel": {}}


class CNPMICoherenceMetric(Metric):

    def __init__(self, flag=True, range=(-1, 1), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        topics = outputData.get_topics()
        dictionary = Dictionary(inputData.texts)
        corpus = [dictionary.doc2bow(doc) for doc in inputData.texts]

        cm = CoherenceModel(topics=topics,
                            texts=inputData.texts,
                            dictionary=dictionary,
                            corpus=corpus,
                            coherence="c_npmi",
                            **self.parameters["coherencemodel"])
        return cm.get_coherence()

    def init_default_parameters(self):
        self.parameters = {"coherencemodel": {}}
