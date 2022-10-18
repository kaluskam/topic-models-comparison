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
    def __init__(self):
        self.texts_topics = None # obiekt, który zawiera teksty i odpowiadające im indeksy tematów
        self.topics = {}
        self.n_topics = 0

    def add_topic(self, topic):
        self.n_topics += 1
        self.topics[self.n_topics] = topic

    def __repr__(self) -> str:
        n_display = min(3, self.n_topics)
        n_skipped = self.n_topics - n_display
        ret_string = ""
        for i in range(1, n_display + 1):
            ret_string += f"Topic {i}\n"
            print(self.topics[i])
            for word, prob in self.topics[i]:
                ret_string += (word + "\t" + str(prob) + "\n")
            ret_string += "\n"
        return ret_string + "... skipped " + str(n_skipped) + " topics"
