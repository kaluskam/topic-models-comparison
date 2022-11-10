import pandas as pd
import os
import datetime as dt

from utils.data_structures import InputData

DATA_DIR = 'data\\preprocessed'


def load_data(subreddits, date_range):
    dfs = []
    for subreddit in subreddits:
        df = pd.read_csv(os.path.join(DATA_DIR, subreddit + '.csv'))
        dfs.append(df.loc[
                          (df['date'] > date_range[0].__str__()) &
                          (df['date'] < date_range[1].__str__()), :])
    final_df = pd.concat(dfs)
    final_df.reset_index(drop=True, inplace=True)

    input_data = InputData(df=final_df)
    input_data.texts_from_df(final_df, 'lematized')

    return input_data



