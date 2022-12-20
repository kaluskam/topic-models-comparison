from metrics.metric import Metric
from octis.evaluation_metrics.similarity_metrics import RBO, WordEmbeddingsPairwiseSimilarity, WordEmbeddingsCentroidSimilarity, WordEmbeddingsWeightedSumSimilarity, PairwiseJaccardSimilarity

class RBOMetric(Metric): #Opposite to InvertedRBO metric
    """A class to calculate Ranked Biased Overlap metric"""
    def __init__(self, flag=True, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "RBO"
        self.description = "Metric calculates average similarity of topic-word lists using Ranked Biased Overlap " \
                           "- a method to compare two ranked lists."
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
    """
    A class to calculate Word Embedding Pairwise Similarity Metric
    """

    def __init__(self, flag=True, range=(-1, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk: top k words on which the topic diversity will be computed,
                word2vec_path: word embedding space in gensim word2vec format,
                binary: If True, indicates whether the data is in binary word2vec format.
        """
        super().__init__(flag, range, parameters)
        self.name = "Word Embedding Pairwise Similarity"
        self.description = "Metric is used to compute the similarity level of meaning of the words inside different topics. Metric calculates average cosine similarity between all of the words in different topics based on " \
                           "embedding model (word2vec-google-news-300 by default)."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        pairwise_similarity = WordEmbeddingsPairwiseSimilarity(**self.parameters["similaritymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return pairwise_similarity.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"similaritymodel": {}}


class WordEmbeddingCentroidSimilarityMetric(Metric):
    """
    A class to calculate Word Embedding Centroid Similarity Metric
    """
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk: top k words on which the topic diversity will be computed,
                word2vec_path: word embedding space in gensim word2vec format,
                binary: If True, indicates whether the data is in binary word2vec format.
        """
        super().__init__(flag, range, parameters)
        self.name = "Word Embedding Centroid Similarity"
        self.description = "Centroid similarity is used to calculate the average distances between topic centers. Metric calculates average vector for each topic based on vectors from embedding model (google-news-300 by default) and then performes cosine similarity " \
                           "on the topic cluster centers."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        centroid_similarity = WordEmbeddingsCentroidSimilarity(**self.parameters["similaritymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return centroid_similarity.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"similaritymodel": {}}

class PairwiseJacckardSimilarityMetric(Metric):
    """
    A class to calculate Pairwise Jacckard Similarity Metric
    """
    def __init__(self, flag=True, range=(0, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk: top k words on which the topic diversity will be computed
        """
        super().__init__(flag, range, parameters)
        self.name = "Pairwise Jacckard Similarity"
        self.description = "Similarity measure based on set operations (union and intersection) on words of each pair of topics."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        pairwise_jacckard = PairwiseJaccardSimilarity(**self.parameters["similaritymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return pairwise_jacckard.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"similaritymodel": {}}



