import pandas as pd
import numpy as np

class InputData:
    """
    Model danych wejściowych do modelu
    """
    def __init__(self, texts = None):
        self.texts = texts # propozycja wstępna, pewnie warto byłoby dodać indeksy dla tych tekstów

    def texts_from_df(self, df, column):
        self.texts = [value[0] for value in df[[column]].values]


class OutputData:
    """
    Model danych wynikowych z modelu
    """
    def __init__(self, documents):
        self.texts_topics = pd.DataFrame({'text_id': [], 'topic_id': []})
        self.documents = documents
        self.topics = []
        self.n_topics = 0
        self.topic_word_matrix = []

    def add_topic(self, words, word_scores, frequency, name="", number_type=None):
        new_topic = Topic(words, word_scores, frequency, name, number_type)
        self.topics.append(new_topic)
        self.n_topics += 1

    def get_topics(self):
        return [topic.words for topic in self.topics]

    def add_texts_topics(self, text_ids, topic_ids):
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
        all_probs = []
        for topic in self.topics:
            probs = topic.word_scores
            all_probs.append(probs)
        topics_word_matrix = np.array(all_probs).astype(float)
        return topics_word_matrix


class Topic:
    """
    an object storing an individual topic
    """

    def __init__(self, words, word_scores, frequency, name="", number_type=None):
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
        return self.words
    
    def get_scores(self):
        return self.word_scores
    
    def get_words_scores(self):
        return self.words, self.word_scores