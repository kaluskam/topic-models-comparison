from utils.data_structures import *
from metrics.metric import Metric

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from octis.evaluation_metrics.coherence_metrics import WECoherencePairwise, WECoherenceCentroid


class UMassCoherenceMetric(Metric):
    """
    A class to calculate UMass Coherence metric.
    """
    def __init__(self, flag=False, range=(-14, 14), parameters=None):
        """
        Parameters
        ----------
        parameters: dict, optional
            dictionary with keys:
                topn (int) : top n words on which the topic diversity will be computed
        """
        super().__init__(flag, range, parameters)
        self.name = "UMass Coherence"
        self.description = "UMass Coherence aims to confirm that the models learned data known to be in the corpus. "
        "The score is defined as a log-conditional-probability calculated on document co-occurences of words in preceeding documents." \
        #"The corpus is used to estimate the probability of a single word as the number of documents in which the word occurs divided by the total number of documents"
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
    """
    A class to calculate CV Coherence metric.
    """
    def __init__(self, flag=True, range=(0, 1), parameters=None):
        """
        Initialize the object of a metric class

        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topn (int) : top n words on which the topic coherence will be computed,
                window_size (int) : the size of the window to be used for coherence measures (default 110)
        """
        super().__init__(flag, range, parameters)
        self.name = "CV Coherence"
        self.description = "CV Coherence is a metric, which indicates how well a topic represents the reference dataset (corpus) " \
        "In this approach the words are compared to the total word set W using two-step measure (m_nlr and cosine similarity). "
        "The corpus is used to determine the word probabilities over a sliding window."
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
    """
    A class to calculate CUCI Coherence metric.
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
            dictionary with keys: topn (int) - top n words on which the topic diversity will be computed,
            window_size (int) - the size of the window to be used for coherence measures using boolean sliding window (default 10)
        """
        super().__init__(flag, range, parameters)
        self.name = "CUCI Coherence"
        self.description = "CUCI Coherence is used to describe how well a topic extracts the information from the corpus. " \
                           "In this approach the pairs of single words are compared using log-ratio measure. " \
                           "The probabilities used in the measure are driven as word co-occurrence counts inside corpus (calculated over a sliding window)."
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
    """
    A class to calculate CNPMI Coherence metric.
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
                topn (int) : top n words on which the topic diversity will be computed
                window_size (int) : the size of the window to be used for coherence measures using boolean sliding window (default 10)
        """
        self.name = "CNPMI Coherence"
        self.description = "CNPMI Coherence is used to assess how well a topic is supported by the corpus. CNPMI metric calculates Normalized Pointwise Mutual Information over pairs of words inside each topic. " \
                           "The calculation is based on probabilities drawn from the reference corpus over a sliding window."
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
    """
    A class to calculate Word Embedding Pairwise Coherence metric.
    """
    def __init__(self, flag=False, range=(-1, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk (int) :  how many most likely words to consider,
                word2vec_path (str) : if word2vec_file is specified retrieves word embeddings file (in word2vec format)
                to compute similarities, otherwise 'word2vec-google-news-300' is downloaded,
                binary (bool) : True if the word2vec file is binary, False otherwise (default False)
        """
        self.name = "WE Pairwise Coherence"
        self.description = "Word Embedding Pairwise Coherence metric is used to determine the quality of the topics based on embeddings. " \
                           "(the default embedding model is word2vec-google-news-300). " \
                           "This metric calculates the average distance between each pair of word vectors inside a topic."
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : topic coherence computed on the word embeddings
                similarities
        """
        super().evaluate(inputData, outputData)
        coherence_pairwise = WECoherencePairwise(**self.parameters["coherencemodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return coherence_pairwise.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"coherencemodel": {}}


class WECoherenceCentroidMetric(Metric):
    """
    A class to calculate Word Embedding Centroid Coherence metric.
    """
    def __init__(self, flag=False, range=(-1, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
            topk (int) :  how many most likely words to consider,
            word2vec_path (str) : if word2vec_file is specified retrieves word embeddings file (in word2vec format)
            to compute similarities, otherwise 'word2vec-google-news-300' is downloaded,
            binary (bool) : True if the word2vec file is binary, False otherwise (default False)
        """
        super().__init__(flag, range, parameters)
        self.name = "WE Centroid Coherence"
        self.description = "Word Embedding Centroid Coherence metric is used to determine the quality of the topics based on embeddings. " \
        "(the default embedding model is word2vec-google-news-300). " \
        "This metric calculates the average distance between each word vector and a topic cluster center."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : topic coherence computed on the word embeddings
                similarities
        """
        super().evaluate(inputData, outputData)
        coherence_centroid = WECoherenceCentroid(**self.parameters["coherencemodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return coherence_centroid.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"coherencemodel": {}}

