from utils.data_structures import InputData
import pickle

class Model:
    """
     Klasa, po której dziedziczą wszystkie używane przez nas modele.
     Każdy z nich musi przeciążać poniższe metody.
    """

    def __init__(self, parameters):
        if parameters is not None:
            assert type(parameters) == dict
        self.parameters = parameters
        self.model = None

    def fit(self, data, n_topics=10):
        assert type(data) == InputData

    def get_output(self):
        pass

    def _match_texts_with_topics(self):
        pass

    def evaluate(self):
        pass

    def choose_number_of_topics(self):
        pass

    def init_default_parameters(self):
        pass

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

