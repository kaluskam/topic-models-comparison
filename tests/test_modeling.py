import copy

import pytest

#from models.berttopic_model import BERTopicModel
from models.lda_model import LDAModel
from models.nmf_model import NMFModel
#from models.berttopic_model import BERTopicModel
from utils.data_structures import InputData
import pandas as pd
import numpy as np
from utils.data_loading import load_downloaded_data
import datetime as dt

@pytest.fixture
def input_data():
   df = load_downloaded_data(['WorldNews'], date_range=[dt.date(2020, 10, 1), dt.date(2020, 11, 1)])
   return df


@pytest.fixture
def sample_output_lda(input_data):
   lda = LDAModel()
   lda.fit(input_data)
   lda.get_output()
   output_lda = lda.output
   return output_lda

@pytest.fixture
def sample_output_nmf(input_data):
   nmf = NMFModel()
   nmf.fit(input_data)
   nmf.get_output()
   output_nmf = nmf.output
   return output_nmf

"""
@pytest.fixture
def sample_output_bertopic(input_data):
   bert = BERTopicModel()
   bert.fit(input_data)
   bert.get_output()
   output_bert = bert.output
   return output_bert
"""

def match_types(output):
   types_match = []
   types_match.append(type(output.texts_topics) == pd.DataFrame)
   types_match.append(output.documents.__class__ == InputData)
   types_match.append(type(output.n_topics) == int)
   types_match.append(type(output.topics_dict) == dict)
   types_match.append(type(output.topic_word_matrix) == np.ndarray)
   return all(types_match) == True

def no_missings(output):
   not_missing = []
   not_missing.append(not output.texts_topics.empty)
   not_missing.append(output.topics_dict != {})
   not_missing.append(output.topic_word_matrix.size != 0)
   return all(not_missing) == True


# Tests
@pytest.mark.parametrize("model,name1, name2", [(LDAModel(), "lda", "num_topics"), (NMFModel(), "nmf", "n_components")])
def test_topics_num(input_data, model, name1, name2):
   model.fit(input_data, n_topics=5)
   output_topics = model.get_output()
   assert model.parameters[name1][name2] == len(output_topics.topics) == output_topics.n_topics


@pytest.mark.parametrize("model,name", [(LDAModel, "lda")])
def test_extremas_filtering(input_data, model, name):
   params = {name: {"num_topics": 10},
             "filter_extremes": {"no_below": 3,  # 20
                                 "no_above": 0.1},  # 0.4
             }
   lda = model(parameters=params)
   lda.fit(input_data)
   assert lda.parameters['filter_extremes']['no_below'] <= min(lda.dictionary.cfs.values())


def test_no_missings_lda(sample_output_lda):
   assert no_missings(sample_output_lda) == True

def test_no_missings_nmf(sample_output_nmf):
   assert no_missings(sample_output_nmf) == True

def test_data_types_lda(sample_output_lda):
   assert match_types(sample_output_lda) == True

def test_data_types_nmf(sample_output_nmf):
   assert match_types(sample_output_nmf) == True

"""
def test_no_missings_bert(sample_output_bertopic):
   assert no_missings(sample_output_bertopic) == True
   
def test_data_types_bert(sample_output_bertopic):
   assert match_types(sample_output_bertopic) == True
"""

