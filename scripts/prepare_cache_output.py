from models.berttopic_model import BERTopicModel
from utils.data_loading import load_downloaded_data, load_cache_output_data, create_output_data_cache_filepath
import datetime as dt
from models.lda_model import LDAModel
from models.nmf_model import NMFModel
import os
import pandas as pd
import definitions as d
import itertools
import numpy as np


def prepare_cache_output(models, subreddits, date_ranges):
    for subreddit in subreddits:
        for date_range in date_ranges:
            for model in models:
                input_data = load_downloaded_data([subreddit], date_range)
                model.fit(input_data)
                output = model.get_output()
                model_alias = str(model).split('.')[2].split('Model')[0].lower()
                filepath = create_output_data_cache_filepath(subreddit, date_range, model_alias)
                output.save(filepath)
                update_dataframe(subreddit, model_alias, date_range)
    return


def update_dataframe(subreddit, model, date_range):
    filepath = os.path.join(d.CACHE_DIR, "cached_files.csv")
    df = pd.DataFrame([[model, subreddit, date_range[0], date_range[1]]], columns=['model', 'subreddit', 'date_range_0', 'date_range_1'])
    if os.path.exists(filepath):
        cache_df = pd.read_csv(filepath)
        cache_df = cache_df.append(df, ignore_index=True)
    else:
        cache_df = df
    cache_df = cache_df.drop_duplicates()
    cache_df.to_csv(filepath, index=False)
    return

#parameters

models = [LDAModel(), NMFModel(), BERTopicModel()]
subreddits = ['recipes']
date_ranges = [[dt.date(2019, 10, 1), dt.date(2020, 9, 30)], [dt.date(2020, 10, 1), dt.date(2021, 9, 30)],
               [dt.date(2021, 10, 1), dt.date(2022, 9, 30)], [dt.date(2019, 10, 1), dt.date(2022, 9, 30)]]

prepare_cache_output(models, subreddits, date_ranges)
