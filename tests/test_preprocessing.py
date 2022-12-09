import os
import pandas as pd
import datetime as dt

import pytest

from utils.preprocessing import DataPreprocessor
import definitions as d


@pytest.fixture
def preprocessor():
    return DataPreprocessor(lematize=True, stem=False, min_word_len=2)

@pytest.fixture
def dataframe():
    df = pd.DataFrame(data = {"lematized": ["THIS IS SOME GROUNDBREAKING TEXT", "Ther3e,:. onc5e'[] was}8{ a/.? do'g", ""]})
    return df


def test_remove_short_words(preprocessor):
    result = preprocessor.remove_short_words(["I", "am", "happy"])
    result2 = preprocessor.remove_short_words(["There", "once", "was", "a", "dog"])
    assert result == ["am", "happy"]
    assert result2 == ["There", "once", "was", "dog"]

def test_remove_digits(preprocessor):
    result = preprocessor.remove_digits(['The1re', 'on4ce', 'was', 'a99', '1d0og10'])
    assert result == ["There", "once", "was", "a", "dog"]


def test_remove_punctuation(preprocessor):
    result = preprocessor.remove_stop_words_and_punctuation(["There,:.", "once'[]", "was}{", "a/.?", "do'g"])
    assert result == ["There", "once", "was", "a", "dog"]

def test_remove_stopwords(preprocessor):
    result = preprocessor.remove_stop_words_and_punctuation(["There", "once", "was", "a", "dog"])
    assert result == ["There", "dog"]

def test_remove_links(preprocessor):
    result = preprocessor.remove_links("You can find it if you search on https://www.google.com")
    assert result == "You can find it if you search on "

def test_expand_contractions(preprocessor):
    result = preprocessor.expand_contractions(["You're", "the", "greatest"])
    assert result == ["You are", "the", "greatest"]

def test_preprocess_text(preprocessor):
    result = preprocessor.preprocess_text("Ther3e,:. onc5e'[] was}8{ a/.? do'g")
    assert result == ["there", "once"]

def test_lower(preprocessor):
    result = preprocessor.preprocess_text("THIS IS SOME GROUNDBREAKING TEXT")
    assert result == ['groundbreaking', "text"]

def test_preprocess_dataframe(preprocessor, dataframe):
    result = preprocessor.preprocess_dataframe(dataframe, "lematized", "preprocessed", False)
    result['preprocessed'] = result['preprocessed'].apply(str)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"lematized": ["THIS IS SOME GROUNDBREAKING TEXT", "Ther3e,:. onc5e'[] was}8{ a/.? do'g", ""],
                                  "preprocessed": ["['groundbreaking', 'text']", "['there', 'once']", "[]"]}))

def test_remove_empty_rows(preprocessor, dataframe):
    result = preprocessor.preprocess_dataframe(dataframe, "lematized", "preprocessed", True)
    result['preprocessed'] = result['preprocessed'].apply(str)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"lematized": ["THIS IS SOME GROUNDBREAKING TEXT", "Ther3e,:. onc5e'[] was}8{ a/.? do'g", ""],
                                  "preprocessed": ["['groundbreaking', 'text']", "['there', 'once']", "[]"]}))