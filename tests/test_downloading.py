import pytest

from utils.downloading import DataDownloader


@pytest.fixture
def subreddit():
    return 'askmen'


@pytest.fixture
def start_date():
    return '2022-10-01'


@pytest.fixture
def end_date():
    return '2022-10-03'


@pytest.fixture
def downloaded_df(subreddit, start_date, end_date):
    dd = DataDownloader(verbose=True)
    df = dd.download_data(subreddit, start_date, end_date, return_df=True, saveas=False)
    return df


def test_column_names(downloaded_df):
    assert len(downloaded_df.columns) == 3
    assert {'title', 'text', 'date'} == set(downloaded_df.columns)


def test_column_types(downloaded_df):
    dtypes = downloaded_df.dtypes
    assert dtypes['title'] == object
    assert dtypes['text'] == object
    assert dtypes['date'] == object


def test_date_range(downloaded_df, start_date, end_date):
    min_date = downloaded_df.date.min()
    max_date = downloaded_df.date.max()
    assert min_date == start_date
    assert max_date == end_date

