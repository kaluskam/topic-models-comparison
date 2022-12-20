import pickle
import pandas as pd
import os

import definitions as d
from utils.data_structures import InputData
from utils.downloading import DataDownloader
from utils.preprocessing import DataPreprocessor
from pathlib import Path


def load_downloaded_data(subreddits, date_range):
    """
    Loads data from already downloaded subreddits for specified date range

    Parameters
    ----------
    subreddits : a list of strings
        subreddit names to be loaded
    date_range: list of DateTime objects
        indicates the first and the last date of a period to be loaded

    Returns
    -------
    input_data : InputData
        merged data for given subreddits and date_range
    """
    dfs = []
    subreddits.sort()

    for subreddit in subreddits:
        preprocessed_df_filepath = os.path.join(d.PREPROCESSED_DIR, subreddit + '.csv')
        if os.path.exists(preprocessed_df_filepath):
            df = pd.read_csv(preprocessed_df_filepath, sep=';')
        else:
            df = load_raw_data_and_preprocess(subreddit)
        dfs.append(df.loc[
                   (df['date'] >= date_range[0].__str__()) &
                   (df['date'] <= date_range[1].__str__()), :])
        final_df = pd.concat(dfs)
        final_df.reset_index(drop=True, inplace=True)
    input_data = InputData(df=final_df)
    input_data.texts_from_df(final_df, 'lematized')
    return input_data


def create_output_data_cache_filepath(subreddit, date_range, model_alias):
    """
    Create an absolute path to the model output file by combining subreddit, date_range and model name

    Parameters
    ----------
    subreddit : str
        subreddit name
    date_range: list of DateTime objects
        indicates the first and the last date of a period to be saved as a cache file
    model_alias: str
        name of the model (lda, nmf, bertopic)

    Returns
    -------
    filepath : str
        absolute path used to save the model output
    """
    filename = str(subreddit) + '&' + str(date_range[0]) + '_' + str(date_range[1]) + '&'+ str(model_alias) +'.obj'
    return os.path.join(d.CACHE_DIR, filename)

def check_cache_existance(subreddit, date_range, model_alias):
    """
    Check if cache file for given parameters exists (model output obtained from a given subreddit, date_range and model)

    Parameters
    ----------
    subreddit : str
        subreddit name
    date_range: list of DateTime objects
        the first and the last date of a posts creation period to be saved as a cache file
    model_alias: str
        name of the model (lda, nmf, bertopic)

    Returns
    -------
    bool
        whether the specified model has been already saved after training on a given subreddit for a chosen date range
    """
    cached_df = pd.read_csv(os.path.join(d.CACHE_DIR, 'cached_files.csv'))
    return not cached_df.loc[(cached_df['subreddit'] == subreddit) & (cached_df['date_range_0'] == date_range[0]) & (cached_df['date_range_1'] == date_range[1]) & (cached_df['model'] == model_alias), :].empty


def load_cache_output_data(filepath):
    """
    Load cached output data from a pickle file

    Parameters
    ----------
    filepath : str
        path to load the model output

    Returns
    -------
    OutputData
        result obtained from training a model
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def load_raw_data_and_preprocess(subreddit):
    """
    Load raw data for a subreddit and perform a preprocessing

    Parameters
    ----------
    subreddit : str
        subreddit to be preprocessed

    Returns
    -------
    df_prep: DataFrame
        preprocessed dataframe which contains lemmatized text from posts and posts creation date
    """
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
    """
    Download a subreddit directly via Pushshift API and preprocess it to obtain lemmatized data

    Parameters
    ----------
    subreddit : str
        subreddit to be downloaded and preprocessed
    date_range: list of DateTime objects
        date ranges of posts creation dates to be downloaded

    Returns
    -------
    df_prep: DataFrame
        preprocessed dataframe which contains lematized text from posts and posts creation date
    """
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


def load_raw_data(subreddits, date_range):
    """
    Load row data, which is needed in data exploration

    Parameters
    ----------
    subreddits : str
        subreddits to be loaded from raw directory data
    date_range: list of DateTime objects
        date ranges of posts to be downloaded

    Returns
    -------
    result_df: DataFrame
        dataframe which contains raw text from posts and posts creation date
    """
    first = True
    subreddits = list(map(lambda x: str(x).lower(), subreddits))
    cwd = Path.cwd()
    for subreddit in subreddits:
        src_path = (cwd / f"./data/raw/{subreddit}.csv").resolve()
        df = pd.read_csv(src_path)
        df = df.sort_values('date')
        df.date = pd.to_datetime(df.date)
        df["raw_text"] = df.title.fillna("") + " " + df.text.fillna("")
        df = df.loc[:, ['raw_text', 'date']]
        df['subreddit'] = subreddit
        if first:
            result_df = df.loc[
                       (df['date'] >= date_range[0].__str__()) &
                       (df['date'] <= date_range[1].__str__()), :]
            first = False
        else:
            result_df = result_df.append(df.loc[
                       (df['date'] >= date_range[0].__str__()) &
                       (df['date'] <= date_range[1].__str__()), :])
    result_df['raw_text'] = result_df['raw_text'].apply(lambda x: DataPreprocessor.remove_links(x))
    return result_df