import pandas as pd
import numpy as np
import re
import os
import pickle


class InputData:
    """
    Class representing input data for model training
    """
    def __init__(self, texts=None, df=None):
        self.texts = texts
        self.df = df

    def texts_from_df(self, df, column):
        """
        Assign preprocessed text from column of a dataframe into field of InputData class

        Parameters
        ----------
        df : DataFrame
            each row contains preprocessed text of a post from subreddit and metadata like post creation date
        column: str
            column name from which the words should be extracted
        """
        if df[column].apply(lambda x: type(x) == str).all():
            self.texts = df[column].apply(lambda x: x.replace('[', '')
                                  .replace(']', '')
                                  .replace("'", '')
                                  .replace(' ', '').split(','))
        else:
            self.texts = [' '.join(value) for value in df[column].values]

    def save(self, filepath):
        """
        Save an InputData object as a pickle file
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)


class OutputData:
    """
    Class representing an output of a model
    """
    def __init__(self, documents):
        self.texts_topics = pd.DataFrame({'text_id': [], 'topic_id': []})
        self.documents = documents
        self.topics = []
        self.topics_dict = {}
        self.n_topics = 0
        self.topic_word_matrix = []

    def add_topic(self, words, word_scores, frequency, name=""):
        """
        Create an instance of a Topic class for each topic and update the topics, topics_dict and n_topics fields for OutputData
        Parameters
        ----------
        words : list of str
            words of which a given topic is consisted
        word_scores: list of np.float32
            probabilities that a certain word belongs to a topic
        frequency: np.float32
            ratio of documents, for which a topic is a most frequent one
        name: str
            name of a topic
        """
        new_topic = Topic(words, word_scores, frequency, name)
        self.topics.append(new_topic)
        self.topics_dict[self.n_topics] = new_topic
        self.n_topics += 1

    def get_topics(self):
        """
        Get all topics

        Returns
        -------
        result: list of str
            words for each topic
        """
        return [topic.words for topic in self.topics]

    def add_texts_topics(self, text_ids, topic_ids):
        """
        Create a dictionary combining id of a text from post with id of a most probable topic

        Parameters
        ----------
        text_ids: list of int
            identifiers of a text (representing post on a subreddit)
        topic_ids: list of int
            identifiers of a most probable topic for a text
        """
        self.texts_topics['text_id'] = text_ids
        self.texts_topics['topic_id'] = topic_ids

    def __repr__(self) -> str:
        n_display = min(3, self.n_topics)
        n_skipped = self.n_topics - n_display
        ret_string = ""
        for i in range(1, n_display + 1):
            ret_string += f"Topic {i}\n"
            ret_string += self.topics[i-1].__repr__()
            ret_string += "\n"
        return ret_string + "... skipped " + str(n_skipped) + " topics"

    def create_topic_word_matrix(self):
        """
        Create a matrix of probabilities for all words inside topics (necessary to calculate for example Log Odds Ratio metric)

        Returns
        ----------
        topic_word_matrix: array of floats
            probabilities of an occurrence of a word in a topic
        """
        all_probs = []
        for topic in self.topics:
            probs = topic.word_scores
            all_probs.append(probs)
        topics_word_matrix = np.array(all_probs).astype(float)
        return topics_word_matrix

    def save(self, filepath):
        """
        Save an OutputData object as a pickle file
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)


class Topic:
    """
    an object storing an individual topic
    """

    def __init__(self, words, word_scores, frequency, name=""):
        """
        Initialize an object of a Topic class
        """
        assert len(words) == len(word_scores)
        self.length = len(words)
        self.words = words
        self.word_scores = word_scores
        self.name = name
        self.frequency = frequency

    def __repr__(self) -> str:
        ret_string = "<\n"
        for i in range(self.length):
                ret_string += (self.words[i] + "\t\t" + str(self.word_scores[i]) + "\n")
        return ret_string + "\n>"

    def get_words(self):
        """
        Returns
        ----------
        words: list of str
            words inside a topic
        """
        return self.words
    
    def get_scores(self):
        """
        Returns
        ----------
        words_scores: list of float
            probabilities that topic contains the assigned words
        """
        return self.word_scores
    
    def get_words_scores(self):
        return self.words, self.word_scores