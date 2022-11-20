import pickle
import pandas as pd
import os

from utils.data_structures import InputData
from utils.downloading import DataDownloader
from utils.preprocessing import DataPreprocessor

CACHE_DIR = 'data\\cache'
PREPROCESSED_DIR = 'data\\preprocessed'
RAW_DIR = 'data\\raw'


def load_data(subreddits, date_range):
    dfs = []
    subreddits.sort()
    input_data_cache_filepath = create_input_data_cache_filepath(subreddits, date_range)
    if os.path.exists(input_data_cache_filepath):
        return load_cache_input_data(input_data_cache_filepath)
    else:
        for subreddit in subreddits:
            preprocessed_df_filepath = os.path.join('..', PREPROCESSED_DIR, subreddit + '.csv')
            if os.path.exists(preprocessed_df_filepath):
                df = pd.read_csv(preprocessed_df_filepath, sep=';')
            else:
                df = get_new_subreddit(subreddit).loc[:, ['lematized', 'date']]
            dfs.append(df.loc[
                       (df['date'] > date_range[0].__str__()) &
                       (df['date'] < date_range[1].__str__()), :])
        final_df = pd.concat(dfs)
        final_df.reset_index(drop=True, inplace=True)

        filepath = input_data_cache_filepath
        input_data = InputData(df=final_df)
        input_data.texts_from_df(final_df, 'lematized')
        input_data.save(filepath)
    return input_data


def create_input_data_cache_filepath(subreddits, date_range):
    filename = '_'.join(subreddits) + '_' + '_'.join(date_range) + '.obj'
    return os.path.join('..', CACHE_DIR, filename)


def load_cache_input_data(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def get_new_subreddit(subreddit):
    filepath = os.path.join('..', RAW_DIR, subreddit + '.csv')
    if os.path.exists(filepath):
        df_raw = pd.read_csv(filepath, sep=',')
    else:
        df_raw = DataDownloader(verbose=True).download_data(subreddit,
                                                            saveas=subreddit)

    df_prep = DataPreprocessor().preprocess_dataframe(df_raw,
                                                      text_column=['title',
                                                                   'text'],
                                                      dest_column='lematized',
                                                      remove_empty_rows=True)
    DataPreprocessor.save(df_prep, subreddit)
    return df_prep


if __name__ == '__main__':
    load_data(['worldnews'], ['2019-10-01', '2022-10-01'])
