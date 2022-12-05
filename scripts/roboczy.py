from train_and_evaluate_models import load_data
from models.nmf_model import NMFModel
from models.lda_model import LDAModel
from models.berttopic_model import BERTopicModel
from utils.preprocessing import DataPreprocessor
from utils.data_structures import InputData
from utils.visualizer import visualise_topics_overtime
from utils.downloading import DataDownloader
import datetime as dt
import pandas as pd

dd = DataDownloader(verbose=True)
df = dd.download_data('pushshift', dt.date(2022, 10, 1), dt.date(2022, 10, 5), return_df=True,
                      saveas=False)