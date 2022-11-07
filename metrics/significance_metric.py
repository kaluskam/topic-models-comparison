from metrics.metric import Metric
from octis.evaluation_metrics.topic_significance_metrics import *

class KLUniformMetric(Metric):
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        kl_uniform = KL_uniform()
        topics_dict = {"topic-word-matrix": outputData.topic_word_matrix}
        return kl_uniform.score(topics_dict)



class KLBackgroundMetric(Metric):
    def __init__(self, flag=False, range=(0, 1), parameters=None):
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        super().evaluate(inputData, outputData)
        kl_background = KL_background()
        topics_dict = {"topic-document-matrix" : np.array(outputData.texts_topics)}
        return kl_background.score(topics_dict)



