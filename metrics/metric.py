from utils.data_structures import InputData, OutputData

class Metric:
    """
    A template class to use for all metrics
    """
    def __init__(self, flag, range, parameters = None):
        if parameters is not None:
            assert type(parameters) == dict
        if flag is not None:
            assert type(flag) == bool
        self.flag = flag
        self.range = range
    
    def evaluate(self, inputData, outputData):
        assert type(inputData) == InputData
        assert type(outputData) == OutputData 

    def init_default_parameters(self):
        pass