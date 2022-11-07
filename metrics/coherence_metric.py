from utils.data_structures import *
from metrics.metric import Metric

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from octis.evaluation_metrics.coherence_metrics import WECoherencePairwise, WECoherenceCentroid


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


class WECoherencePairwiseMetric(Metric):
    def __init__(self, flag=False, range=(-1, 1), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        coherence_pairwise = WECoherencePairwise(**self.parameters["coherencemodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return coherence_pairwise.score(topics_dict)

    def init_default_parameters(self):
        """
        Parameters
        ----------
        dictionary with keys
        topk : how many most likely words to consider
        word2vec_path : if word2vec_file is specified retrieves word embeddings file (in word2vec format)
        to compute similarities, otherwise 'word2vec-google-news-300' is downloaded
        binary : True if the word2vec file is binary, False otherwise (default False)
        """
        self.parameters = {"coherencemodel": {}}


class WECoherenceCentroidMetric(Metric):
    def __init__(self, flag=False, range=(-1, 1), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        coherence_centroid = WECoherenceCentroid(**self.parameters["coherencemodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return coherence_centroid.score(topics_dict)

    def init_default_parameters(self):
        """
        Parameters
        ----------
        dictionary with keys
        topk : how many most likely words to consider
        word2vec_path : if word2vec_file is specified retrieves word embeddings file (in word2vec format)
        to compute similarities, otherwise 'word2vec-google-news-300' is downloaded
        binary : True if the word2vec file is binary, False otherwise (default False)
        """
        self.parameters = {"coherencemodel": {}}

