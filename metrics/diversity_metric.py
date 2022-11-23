from utils.data_structures import *
from metrics.metric import Metric
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO, LogOddsRatio, WordEmbeddingsInvertedRBO, WordEmbeddingsInvertedRBOCentroid, KLDivergence
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import re

class TopicDiversityMetric(Metric):
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        self.name = "Topic Diversity"
        self.description = "Topic Diveristy calculates the ratio of diverse words used to describe the topics."
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        td = TopicDiversity(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return td.score(topics_dict)

    def init_default_parameters(self):
        """
        :param topk: top k words on which the topic diversity will be computed
        """
        self.parameters = {"diversitymodel": {}}


class InvertedRBOMetric(Metric):
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "Inverted RBO"
        self.description = "Metric calculates average diversity of topic-word lists using Inverted Ranked Biased Overlap " \
                           "- a method to compare two ranked lists."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        invrbo = InvertedRBO(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return invrbo.score(topics_dict)

    def init_default_parameters(self):
        """
        Initialize metric Inverted Ranked-Biased Overlap
        :param topk: top k words on which the topic diversity will be computed
        :param weight: weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (default 0.9)
        """
        self.parameters = {"diversitymodel": {}}

class LogOddsRatioMetric(Metric):
    def __init__(self, flag=False, range=(0, float("inf"))):
        self.name = "Log Odds Ratio"
        self.description = "Log Odds Ratio compares the usage of a word across " \
                           "different documents. The metric is calculated based on a topic-word probability matrix."
        super().__init__(flag, range)

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        log_odds_ratio = LogOddsRatio()
        topics_word_dict = {"topic-word-matrix": outputData.topic_word_matrix}
        print(outputData.topic_word_matrix)
        return log_odds_ratio.score(topics_word_dict)

class WordEmbeddingsInvertedRBOMetric(Metric):
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        self.name = "Word Embeddings Inverted RBO"
        self.description = "Metric calculates average pairwise diversity of topic-word vector lists using Inverted Ranked Biased Overlap and embedding model (the default embedding model is word2vec-google-news-300)."
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        we_inv_rbo = WordEmbeddingsInvertedRBO(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return we_inv_rbo.score(topics_dict)

    def init_default_parameters(self):
        """
        Initialize metric WE-IRBO-Match
        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :param word2vec_path: word embedding space in gensim word2vec format
        :param weight: Weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (Default 0.9)
        :param normalize: if true, normalize the cosine similarity
        """
        self.parameters = {"diversitymodel": {}}


class WordEmbeddingsInvertedRBOCentroidMetric(Metric):
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "Word Embeddings Inverted RBO Centroid"
        self.description = "Metric calculates average diversity of topic-word vector lists using Inverted Ranked Biased Overlap and embedding model (the default embedding model is word2vec-google-news-300). " \
                           "The diversity is calculated between each word vector list and mean of word vectors for each topic."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        we_inv_rbo = WordEmbeddingsInvertedRBOCentroid(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return we_inv_rbo.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"diversitymodel": {}}



