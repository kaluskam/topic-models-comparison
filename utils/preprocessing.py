import string
import contractions
import re
import nltk
import copy
import pandas as pd
from utils.data_structures import InputData
import os
import definitions as d

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer


STOP_WORDS = stopwords.words('english')
PUNCTUATION = list(string.punctuation)
STOP_WORDS_PUNCT = set(STOP_WORDS + PUNCTUATION)
PUNCTUATION_DICT = dict.fromkeys(string.punctuation, '')


class DataPreprocessor:
    """
    Class to preprocess raw text from a dataframe
    """
    def __init__(self, lematize=True, stem=False, min_word_len=2):
        self.lematize = lematize
        self.stem = stem
        self.min_word_len = min_word_len

    def preprocess_dataframe(self, df, text_column, dest_column, remove_empty_rows=True, inplace=False):
        """
        Parameters
        ----------
        df : DataFrame
            dataframe containing posts from a subreddit
        text_column: str
            column name from which the text will be preprocessed
        dest_column: str
            destination column for preprocessed text
        remove_empty_rows: bool
            indicates whether to remove empty rows
        inplace: bool
            indicates whether to perform preprocessing on input dataframe or a copy

        Returns
        -------
        df_copy = DataFrame
            dataframe containing column with preprocessed text
        """
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
            df_copy = df_copy[df_copy[dest_column].apply(lambda x: type(x) == list)]
            df_copy = df_copy[df_copy[dest_column].apply(lambda x: len(x)) > 0]
        return df_copy

    def preprocess_text(self, text):
        """
        Method to conduct all steps of text preprocessing: links removal, keeping only ASCII symbols,
        stop words removal, punctuation removal etc.

        Parameters
        ----------
        text : str
            text from a subreddit post

        Returns
        -------
        words : list of str
            words after preprocessing
        """
        text = DataPreprocessor.remove_links(text)
        text = DataPreprocessor.keep_ascii_only(text)
        words = contractions.fix(text.lower())
        words = word_tokenize(words)
        words = DataPreprocessor.remove_stop_words_and_punctuation(words)
        words = DataPreprocessor.remove_digits(words)
        if self.stem:
            words = [SnowballStemmer('english').stem(word) for word in words]
        if self.lematize:
            words = [WordNetLemmatizer().lemmatize(word, pos="v") for word in words]

        return self.remove_short_words(words)

    def remove_short_words(self, words):
        return [word for word in words if len(word) >= self.min_word_len]

    def paste_columns(self, df_copy, text_column):
        """
        Method to paste the text from different columns, usually title and text of a post

        Parameters
        ----------
        df_copy : DataFrame
            reference dataframe
        text_column: list of str
            columns to be pasted together

        Returns
        -------
        initial_text: list
            initial text merged with the text from specified columns
        """
        initial_text = df_copy[text_column[0]]
        for i in range(len(text_column) - 1):
            df_copy[text_column[i + 1]] = df_copy[text_column[i + 1]].apply(
                lambda text: text if type(text) == str else '')
            next_text = df_copy[text_column[i + 1]]
            initial_text = initial_text + " " + next_text
        return initial_text

    def read_data(self, subreddit):
        """
        Read the dataframe from raw directory by specifying the subreddit name
        """
        file = subreddit.lower()
        df = pd.read_csv(os.path.join(d.RAW_DIR, file) + '.csv')
        return df

    @staticmethod
    def save(df, subreddit):
        path = d.PREPROCESSED_DIR
        if not os.path.exists(path):
            os.mkdir(path)
        df.to_csv(os.path.join(path, subreddit.lower()) + '.csv', index=False, sep=';')

    @staticmethod
    def to_InputDataModel(df, text_column):
        return InputData(texts=df[text_column])

    @staticmethod
    def expand_contractions(words):
        return [contractions.fix(word) for word in words]

    @staticmethod
    def remove_stop_words_and_punctuation(words):
        words = [word for word in words if word not in STOP_WORDS]
        return [word.translate(str.maketrans(PUNCTUATION_DICT)) for word in words]

    @staticmethod
    def remove_digits(words):
        return [re.sub('\d+', '', word) for word in words]

    @staticmethod
    def remove_links(text):
        text = re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)", "", text, count=0, flags=0)
        return text

    @staticmethod
    def keep_ascii_only(text):
        return text.encode('ascii', errors='ignore').decode()