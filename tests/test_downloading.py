import os
import datetime as dt

import pytest

from utils.downloading import DataDownloader
import definitions as d


@pytest.fixture
def subreddit():
    return 'pushshift'


@pytest.fixture
def start_date():
    return dt.date(2022, 10, 1)


@pytest.fixture
def end_date():
    return dt.date(2022, 10, 5)


@pytest.fixture
def downloaded_df(subreddit, start_date, end_date):
    dd = DataDownloader(verbose=True)
    df = dd.download_data(subreddit, start_date, end_date, return_df=True, saveas=False)
    return df


# @pytest.mark.special
# def test_file_exists(subreddit):
#     filepath = os.path.join(d.RAW_DIR, subreddit + '.csv')
#     print(filepath)
#     assert os.path.exists(filepath)


def test_column_names(downloaded_df):
    assert len(downloaded_df.columns) == 3
    assert {'title', 'text', 'date'} == set(downloaded_df.columns)


def test_column_types(downloaded_df):
    dtypes = downloaded_df.dtypes
    assert dtypes['title'] == str
    assert dtypes['text'] == str
    assert dtypes['date'] == str


def test_date_range(downloaded_df, start_date, end_date):
    min_date = downloaded_df.date.min()
    max_date = downloaded_df.date.max()
    assert min_date == start_date
    assert max_date == end_date

#
# def test_downloading(subreddit='askdocs', start_date=dt.date(2022, 10, 1),
#                      end_date=dt.date(2022, 10, 5)):
#     assert_file_exists(subreddit)
#     check_column_names(df)
#     check_column_types(df)
#     check_date_range(df, start_date, end_date)

# if __name__ == '__main__':
#     import pytest
#     import definitions as d
#     import os
#     import datetime as dt
#     from utils.downloading import DataDownloader
#     test_downloading('askdocs', dt.date(2022, 10, 1), dt.date(2022, 10, 5))
#
