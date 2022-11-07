import string
import contractions
import re
import nltk
import copy
import pandas as pd
from utils.data_structures import InputData
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

STOP_WORDS = stopwords.words('english')
PUNCTUATION = list(string.punctuation) + ["\"\"", "``", "\'\'"]
STOP_WORDS_PUNCT = set(STOP_WORDS + PUNCTUATION)


class DataPreprocessor:
    def __init__(self, lematize=False, stem=True, min_word_len=2):
        self.lematize = lematize
        self.stem = stem
        self.min_word_len = min_word_len

    def preprocess_dataframe(self, df, text_column, dest_column, remove_empty_rows, inplace=False):
        if inplace:
            df_copy = df
        else:
            df_copy = copy.deepcopy(df)

        if type(text_column) == str:
            text_column = [text_column]

        df_copy[text_column[0]] = df_copy[text_column[0]].apply(lambda text: text if type(text) == str else '')
        initial_text = df_copy[text_column[0]]
        if type(text_column) == list and len(text_column) != 1:
            initial_text = self.paste_columns(df_copy, text_column)

        df_copy[dest_column] = initial_text.apply(lambda text: self.preprocess_text(text)) #df_copy[text_column[0]]
        if remove_empty_rows:
            df_copy = df_copy[df_copy[dest_column] != '']
        return df_copy

    def preprocess_text(self, text):
        words = contractions.fix(text.lower())
        words = word_tokenize(words)
        if self.stem:
            words = [SnowballStemmer('english').stem(word) for word in words]
        if self.lematize:
            words = [WordNetLemmatizer().lemmatize(word) for word in words]

        words = DataPreprocessor.remove_stop_words_and_punctuation(words)
        words = DataPreprocessor.remove_digits(words)
        return self.remove_short_words(words)

    def remove_short_words(self, words):
        return [word for word in words if len(word) >= self.min_word_len]

    def paste_columns(self, df_copy, text_column):
        initial_text = df_copy[text_column[0]]
        for i in range(len(text_column) - 1):
            df_copy[text_column[i + 1]] = df_copy[text_column[i + 1]].apply(
                lambda text: text if type(text) == str else '')
            next_text = df_copy[text_column[i + 1]]
            initial_text = initial_text + " " + next_text
        return initial_text

    def read_data(self, subreddit):
        file = subreddit.lower()
        df = pd.read_csv('../data/raw/' + file + '.csv')
        return df

    def save_data(self, df, subreddit):
        path = "../data/preprocessed/"
        if not os.path.exists(path):
            os.mkdir(path)
        df.to_csv(path + subreddit.lower() + '.csv', index=False)
        return


    @staticmethod
    def to_InputDataModel(df, text_column):
        return InputData(texts=df[text_column])

    @staticmethod
    def expand_contractions(words):
        return [contractions.fix(word) for word in words]

    @staticmethod
    def remove_stop_words_and_punctuation(words):
        return [word for word in words if word not in STOP_WORDS_PUNCT]

    @staticmethod
    def remove_digits(words):
        return [re.sub('\d+', '', word) for word in words]