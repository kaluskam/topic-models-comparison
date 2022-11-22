import os
import datetime as dt


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PREPROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'preprocessed')
CACHE_DIR = os.path.join(ROOT_DIR, 'data', 'cache')

START_DATE = dt.date(2019, 10, 1)
END_DATE = dt.date(2022, 9, 30)