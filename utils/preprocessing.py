import string
import contractions
import re
import nltk
import copy

from data_structures import InputData

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

STOP_WORDS = stopwords.words('english')
PUNCTUATION = list(string.punctuation)
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

        df_copy[text_column] = df_copy[text_column].apply(lambda text: text if type(text) == str else '')
        df_copy[dest_column] = df_copy[text_column].apply(lambda text: self.preprocess_text(text))
        if remove_empty_rows:
            df_copy = df_copy[df_copy[dest_column] != '']
        return df_copy

    def preprocess_text(self, text):
        words = word_tokenize(text.lower())
        if self.stem:
            words = [SnowballStemmer('english').stem(word) for word in words]
        if self.lematize:
            words = [WordNetLemmatizer().lemmatize(word) for word in words]

        words = DataPreprocessor.expand_contractions(words)
        words = DataPreprocessor.remove_stop_words_and_punctuation(words)
        words = DataPreprocessor.remove_digits(words)
        return self.remove_short_words(words)

    def remove_short_words(self, words):
        return [word for word in words if len(word) >= self.min_word_len]

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