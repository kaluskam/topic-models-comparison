from train_and_evaluate_models import load_data
from models.nmf_model import NMFModel
from utils.preprocessing import DataPreprocessor
from utils.data_structures import InputData
from utils.visualizer import visualise_topics_overtime

import pandas as pd
df = pd.read_csv("..\\data\\askmen_df.csv")
dp = DataPreprocessor(True, False)
text_df = dp.preprocess_dataframe(df.loc[1:10000, ], "selftext",
                                  "processed_text", True, title_column="title")
datamodel = InputData()
datamodel.texts_from_df(text_df, "processed_text")

nmf = NMFModel()
nmf.fit(datamodel)
nmf_topics = nmf.get_topics()
nmf.match_texts_with_topics()
print(nmf.output.texts_topics)
texts_topics_df = nmf.output.texts_topics


r = pd.merge(text_df, texts_topics_df, left_index=True, right_on='text_id')
visualise_topics_overtime(r, 'created', nmf.output)