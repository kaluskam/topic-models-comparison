from train_and_evaluate_models import load_data
from models.nmf_model import NMFModel
from models.lda_model import LDAModel
from models.berttopic_model import BERTopicModel
from utils.preprocessing import DataPreprocessor
from utils.data_structures import InputData
from utils.visualizer import visualise_topics_overtime

import pandas as pd
df = pd.read_csv("..\\data\\preprocessed\\askmen.csv")
dp = DataPreprocessor(True, False)
# text_df = dp.preprocess_dataframe(df.loc[1:1000, ], "selftext",
#                                   "processed_text", True, title_column="title")
datamodel = InputData()
datamodel.texts_from_df(df, "lematized")

nmf = NMFModel()
nmf.fit(datamodel)
nmf_topics = nmf.get_output()
# print(nmf.output.texts_topics)
texts_topics_df = nmf.output.texts_topics


r = pd.merge(df, texts_topics_df, left_index=True, right_on='text_id')
visualise_topics_overtime(r, 'date', nmf.output, 'day')