from metrics.metric import Metric
from octis.evaluation_metrics.similarity_metrics import RBO, WordEmbeddingsPairwiseSimilarity, WordEmbeddingsCentroidSimilarity, WordEmbeddingsWeightedSumSimilarity, PairwiseJaccardSimilarity

class RBOMetric(Metric): #Opposite to InvertedRBO metric
    def __init__(self, flag=True, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "RBO"
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        rbo = RBO(**self.parameters["similaritymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return rbo.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"similaritymodel": {}}

class WordEmbeddingPairwiseSimilarityMetric(Metric):

    def __init__(self, flag=True, range=(-1, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "Word Embedding Pairwise Similarity"
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        pairwise_similarity = WordEmbeddingsPairwiseSimilarity(**self.parameters["similaritymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return pairwise_similarity.score(topics_dict)

    def init_default_parameters(self):
        """
        Initialize metric WE pairwise similarity
        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :param word2vec_path: word embedding space in gensim word2vec format
        :param binary: If True, indicates whether the data is in binary word2vec format.
        """
        self.parameters = {"similaritymodel": {}}


class WordEmbeddingCentroidSimilarityMetric(Metric):
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "Word Embedding Centroid Similarity"
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        centroid_similarity = WordEmbeddingsCentroidSimilarity(**self.parameters["similaritymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return centroid_similarity.score(topics_dict)

    def init_default_parameters(self):
        """
       Initialize metric WE centroid similarity
       Parameters
       ----------
       :param topk: top k words on which the topic diversity will be computed
       :param word2vec_path: word embedding space in gensim word2vec format
       :param binary: If True, indicates whether the data is in binary word2vec format.
       """
        self.parameters = {"similaritymodel": {}}

class PairwiseJacckardSimilarityMetric(Metric):
    def __init__(self, flag=True, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "Pairwise Jacckard Similarity"
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        pairwise_jacckard = PairwiseJaccardSimilarity(**self.parameters["similaritymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return pairwise_jacckard.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"similaritymodel": {}}



