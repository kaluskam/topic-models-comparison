import pickle
import pandas as pd
import os

import definitions as d
from utils.data_structures import InputData
from utils.downloading import DataDownloader
from utils.preprocessing import DataPreprocessor


def load_downloaded_data(subreddits, date_range):
    dfs = []
    subreddits.sort()
    input_data_cache_filepath = create_input_data_cache_filepath(subreddits, date_range)
    if os.path.exists(input_data_cache_filepath):
        return load_cache_input_data(input_data_cache_filepath)
    else:
        for subreddit in subreddits:
            preprocessed_df_filepath = os.path.join(d.PREPROCESSED_DIR, subreddit + '.csv')
            if os.path.exists(preprocessed_df_filepath):
                df = pd.read_csv(preprocessed_df_filepath, sep=';')
                for col in df.columns:
                    if col != 'date':
                        df[col] = df[col].apply(lambda x: str(x).split(', '))
            else:
                df = load_raw_data_and_preprocess(subreddit)
            dfs.append(df.loc[
                       (df['date'] >= date_range[0].__str__()) &
                       (df['date'] <= date_range[1].__str__()), :])
        final_df = pd.concat(dfs)
        final_df.reset_index(drop=True, inplace=True)

        filepath = input_data_cache_filepath
        input_data = InputData(df=final_df)
        input_data.texts_from_df(final_df, 'lematized')
        input_data.save(filepath)
    return input_data


def create_input_data_cache_filepath(subreddits, date_range):
    filename = '_'.join(subreddits) + '&' + str(date_range[0]) + '_' + str(date_range[1]) + '.obj'
    return os.path.join(d.CACHE_DIR, filename)


def load_cache_input_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def load_raw_data_and_preprocess(subreddit):
    filepath = os.path.join(d.RAW_DIR, subreddit + '.csv')
    df_raw = pd.read_csv(filepath, sep=',')
    df_prep = DataPreprocessor().preprocess_dataframe(df_raw,
                                                      text_column=['title',
                                                                   'text'],
                                                      dest_column='lematized',
                                                      remove_empty_rows=True)
    df_prep = df_prep.loc[:, ['lematized', 'date']]
    DataPreprocessor.save(df_prep, subreddit)
    return df_prep


def download_new_subreddit_and_preprocess(subreddit, date_range):
    df_raw = DataDownloader(verbose=True).download_data(subreddit,
                                                        start_date=str(date_range[0]),
                                                        end_date=str(date_range[1]),
                                                        saveas=subreddit)
    df_prep = DataPreprocessor().preprocess_dataframe(df_raw,
                                                      text_column=['title',
                                                                   'text'],
                                                      dest_column='lematized',
                                                      remove_empty_rows=True)
    df_prep = df_prep.loc[:, ['lematized', 'date']]
    DataPreprocessor.save(df_prep, subreddit)
    return df_prep
