import os
import datetime as dt
import pandas as pd
import pytest

from metrics.coherence_metric import *
from metrics.diversity_metric import *
from metrics.significance_metric import *
from metrics.similarity_metric import *
import definitions as d


model_names = ["NMF", "LDA"]
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
    InvertedRBOMetric(),
    LogOddsRatioMetric(),
    WordEmbeddingsInvertedRBOMetric(),
    WordEmbeddingsInvertedRBOCentroidMetric()
]

metric_names = [metric.name for metric in metrics]

@pytest.fixture
def results_dataframe():
    return pd.read_csv("scores_df.csv", index_col = 0) ##temporary, musi się liczyć samo potem


@pytest.mark.parametrize("metric, metricname", [(metric, metric_name) for (metric, metric_name) in zip(metrics, metric_names)])
def test_metric_has_name(metric, metricname):
    assert metric.name is not None
    assert type(metric.description) == str

@pytest.mark.parametrize("metric, metricname", [(metric, metric_name) for (metric, metric_name) in zip(metrics, metric_names)])
def test_metric_has_description(metric, metricname):
    assert metric.description is not None
    assert type(metric.description) == str

@pytest.mark.parametrize("metric, metricname", [(metric, metric_name) for (metric, metric_name) in zip(metrics, metric_names)])
def test_metric_has_range(metric, metricname):
    assert metric.range is not None
    assert type(metric.range) == tuple

@pytest.mark.parametrize("metric, metricname", [(metric, metric_name) for (metric, metric_name) in zip(metrics, metric_names)])
def test_metric_has_flag(metric, metricname):
    assert metric.flag is not None
    assert type(metric.flag) == bool

@pytest.mark.parametrize("metric, metricname", [(metric, metric_name) for (metric, metric_name) in zip(metrics, metric_names)])
@pytest.mark.parametrize("model", model_names)
def test_metric_in_range(metric, metricname, model, results_dataframe):
    lower_bound = metric.range[0]
    upper_bound = metric.range[1]
    rowname = model
    colname = metricname
    assert results_dataframe.loc[rowname, colname] <= upper_bound
    assert results_dataframe.loc[rowname, colname] >= lower_bound




