import os
import datetime as dt


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PREPROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'preprocessed')
CACHE_DIR = os.path.join(ROOT_DIR, 'output', 'cache')
GOOGLE_TRENDS_DIR = os.path.join(ROOT_DIR, 'google_trends')
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')

START_DATE = dt.date(2019, 10, 1)
END_DATE = dt.date(2022, 9, 30)

ALERT_MESSAGE = 'Plots showed below are only exemplary. If you want them to reflect selected options,\n please click the "Run analysis" button.'
ALERT_TITLE = 'Please click "Run analysis" button'