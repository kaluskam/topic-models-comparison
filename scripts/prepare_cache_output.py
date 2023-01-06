from models.berttopic_model import BERTopicModel
from utils.data_loading import load_downloaded_data, load_cache_output_data, create_output_data_cache_filepath
import datetime as dt
from models.lda_model import LDAModel
from models.nmf_model import NMFModel
import os
import pandas as pd
import hdbscan
import definitions as d


from metrics.coherence_metric import *
from metrics.diversity_metric import *
from metrics.significance_metric import *
from metrics.similarity_metric import *

metrics = [
    KLUniformMetric(),
    KLBackgroundMetric(),
    RBOMetric(),
    WordEmbeddingPairwiseSimilarityMetric(),
    WordEmbeddingCentroidSimilarityMetric(),
    PairwiseJacckardSimilarityMetric(),
    UMassCoherenceMetric(),
    CVCoherenceMetric(),
    CUCICoherenceMetric(),
    CNPMICoherenceMetric(),
    WECoherencePairwiseMetric(),
    WECoherenceCentroidMetric(),
    TopicDiversityMetric(),
    LogOddsRatioMetric()
]


def prepare_cache_output(models, subreddits, date_ranges, num_topics):
    for subreddit in subreddits:
        print(subreddit)
        for date_range in date_ranges:
            print(date_range)
            for model in models:
                print(model)
                input_data = load_downloaded_data([subreddit], date_range)
                model.fit(input_data, num_topics)
                output = model.get_output()
                model_alias = str(model).split('.')[2].split('Model')[0].lower() + "2"
                num_topics_upd = "None" if model_alias == "bertopic" else num_topics
                filepath = create_output_data_cache_filepath(subreddit, date_range, model_alias, num_topics_upd)
                print("calculating metrics")
                output.calculate_metrics(metrics)
                output.save(filepath)
                update_dataframe(subreddit, model_alias, date_range, num_topics_upd)
    return


def update_dataframe(subreddit, model, date_range, num_topics):
    filepath = os.path.join(d.CACHE_DIR, "cached_files.csv")
    df = pd.DataFrame([[model, subreddit, date_range[0], date_range[1], num_topics]], columns=['model', 'subreddit', 'date_range_0', 'date_range_1', "num_topics"])
    if os.path.exists(filepath):
        cache_df = pd.read_csv(filepath)
        cache_df = cache_df.append(df, ignore_index=True)
    else:
        cache_df = df
    cache_df = cache_df.drop_duplicates()
    cache_df.to_csv(filepath, index=False)
    return


#parameters
clusterer = hdbscan.HDBSCAN(algorithm='best', approx_min_span_tree=True,
                                    gen_min_span_tree=False, leaf_size=30, metric='euclidean', min_cluster_size=1000,
                                    min_samples=20, cluster_selection_epsilon=0.2, p=None)
parameters = {"bertopic": {"n_gram_range": (1, 1),
                                "top_n_words": 10,
                                "language": "english",
                                "hdbscan_model": clusterer,
                                #"min_topic_size": 100,
                                "nr_topics": 20}}

models = [BERTopicModel(parameters=parameters)]
subreddits = ['worldnews']
date_ranges = [[dt.date(2019, 10, 1), dt.date(2022, 9, 30)]]

prepare_cache_output(models, subreddits, date_ranges, num_topics=20)
