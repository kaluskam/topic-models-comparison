from utils.data_structures import InputData
import pickle

class Model:
    """
     Class, which represents a model. Each model should inherit from it and override the
     following methods.
    """

    def __init__(self, parameters):
        """
        Initialize the instance of a model class with passed parameters

        Parameters
        ----------
        parameters : dictionary
            parameters passed to the model
        """
        if parameters is not None:
            assert type(parameters) == dict
        self.parameters = parameters
        self.model = None

    def fit(self, data, n_topics=10):
        """
        Fit the model

        Parameters
        ----------
        data : InputData
            contains text from documents
        n_topics: int
            number of topics selected by a model, 10 by default
        """
        assert type(data) == InputData

    def get_output(self):
        """
        Assign output of the model to the instance of an OutputData class
        """
        pass

    def _match_texts_with_topics(self):
        """
        Match each document with the topic of the highest probability
        """
        pass

    def init_default_parameters(self):
        pass

    def save(self, filepath):
        """
        A method to save the model as a pickle file

        Parameters
        ----------
        filepath : str
            a path to save the model
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

