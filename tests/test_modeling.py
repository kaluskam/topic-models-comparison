import copy

import pytest
import itertools
from models.lda_model import LDAModel
from models.nmf_model import NMFModel
from models.berttopic_model import BERTopicModel
from utils.data_structures import InputData
import pandas as pd
import numpy as np
from utils.data_loading import load_downloaded_data
import datetime as dt
import definitions as d
import os
import re


@pytest.fixture(scope="session")
def input_data():
    input_df = pd.read_csv(os.path.join(d.PREPROCESSED_DIR, "worldnews.csv"), sep =";")
    for col in input_df.columns:
                if col != 'date':
                    input_df[col] = input_df[col].apply(lambda x: re.sub('[\'\"\[\],]', '', str(x)))
    input_df = input_df.head(80)
    input_data = InputData()
    input_data.texts_from_df(input_df, "lematized")
    return input_data

@pytest.fixture(scope="session")
def sample_output_lda(input_data):
    lda = LDAModel()
    lda.fit(input_data)
    lda.get_output()
    output_lda = lda.output
    return output_lda

@pytest.fixture(scope="session")
def sample_output_nmf(input_data):
    nmf = NMFModel()
    nmf.fit(input_data)
    nmf.get_output()
    output_nmf = nmf.output
    return output_nmf

@pytest.fixture(scope="session")
def sample_output_bertopic(input_data):
   bert = BERTopicModel()
   bert.fit(input_data)
   bert.get_output()
   output_bert = bert.output
   return output_bert

@pytest.fixture(scope="session")
def sample_output(sample_output_lda, sample_output_nmf, sample_output_bertopic):
    return {"LDA": sample_output_lda, "NMF": sample_output_nmf, "BERTopic": sample_output_bertopic}

# Tests
@pytest.mark.parametrize("model,name1, name2",
                         [(LDAModel(), "lda", "num_topics"),
                          (NMFModel(), "nmf", "n_components")])
def test_topics_num(input_data, model, name1, name2):
    model.fit(input_data, n_topics=5)
    output_topics = model.get_output()
    assert model.parameters[name1][name2] == len(
        output_topics.topics) == output_topics.n_topics


@pytest.mark.parametrize("model,name", [(LDAModel, "lda")])
def test_extremas_filtering(input_data, model, name):
    params = {name: {"num_topics": 10},
              "filter_extremes": {"no_below": 3,  # 20
                                  "no_above": 0.1},  # 0.4
              }
    lda = model(parameters=params)
    lda.fit(input_data)
    assert lda.parameters['filter_extremes']['no_below'] <= min(
        lda.dictionary.cfs.values())



@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_matching_text_topic(sample_output, model_name):
    assert not sample_output[model_name].texts_topics.empty

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_if_topics_found(sample_output, model_name):
    assert sample_output[model_name].topics != []

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_size_topic_matrix(sample_output, model_name):
    assert sample_output[model_name].topic_word_matrix.size != 0

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_text_topics_type(sample_output, model_name):
    assert type(sample_output[model_name].texts_topics) == pd.DataFrame

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_documents_type(sample_output, model_name):
    assert sample_output[model_name].documents.__class__ == InputData

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_n_topics_type(sample_output, model_name):
    assert type(sample_output[model_name].n_topics) == int

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_n_topics_type(sample_output, model_name):
    assert type(sample_output[model_name].n_topics) == int

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_topics_dict_type(sample_output, model_name):
    assert type(sample_output[model_name].n_topics) == dict

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_topics_dict_type(sample_output, model_name):
    assert type(sample_output[model_name].topic_word_matrix) == np.ndarray

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_topics_prob_range(sample_output, model_name):
    assert (np.min(sample_output[model_name].topic_word_matrix) > 0) & (np.max(sample_output[model_name].topic_word_matrix) < 1)

@pytest.mark.parametrize("model_name", ["LDA", "NMF", "BERTopic"])
def test_topics_contain_letters_only(sample_output, model_name):
    all_words = [topic.words for topic in sample_output[model_name].topics]
    all_words = list(itertools.chain(*all_words))
    assert all(list(map(lambda x: str.isalpha(x), all_words)))


