from metrics.metric import Metric
from octis.evaluation_metrics.topic_significance_metrics import *

class KLUniformMetric(Metric):
    """
    A class to calculate Kullback-Leiber Unifrom metric.
    """
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        self.name = "KL Uniform"
        self.description = "Metric is used to measure the significance of a given topic by checking how close is the topic-word " \
                           "probability distribution to the unifrom distribution. In this purpose the Kullback-Leiber divergence is calculated."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : Kullback-Leiber distance calculated on topic-word probability matrix
        """
        super().evaluate(inputData, outputData)
        kl_uniform = KL_uniform()
        topics_dict = {"topic-word-matrix": outputData.topic_word_matrix}
        return kl_uniform.score(topics_dict)

class KLBackgroundMetric(Metric):
    """
    A class to calculate Kullback-Leiber Background metric.
    """
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        self.name = "KL Background"
        self.description = "Kl Background metric is used to investigate the distribution of topics over documents using Kullback-Leiber distance. " \
                           "If a topic distribution over document is close to uniform, this topic is treated as insignificant."
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : Kullback-Leiber distance calculated on topic-document probability matrix
        """
        super().evaluate(inputData, outputData)
        kl_background = KL_background()
        topics_dict = {"topic-document-matrix" : np.array(outputData.texts_topics)}
        return kl_background.score(topics_dict)



