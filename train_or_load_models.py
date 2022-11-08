from models.nmf_model import NMFModel
from models.lda_model import LDAModel
from models.berttopic_model import BERTopicModel
from utils.preprocessing import DataPreprocessor
from utils.data_structures import InputData
from utils.visualizer import visualise_topics_overtime
from os.path import exists

from metrics.coherence_metric import *
from metrics.diversity_metric import *
from metrics.significance_metric import *
from metrics.similarity_metric import *


import pandas as pd
import pickle

def load_data():
    if exists("inputdata.obj"):
        with open("inputdata.obj", "rb") as f:
            inputdata = pickle.load(f)
    else:
        df = pd.read_csv("data/askmen_df.csv")
        dp = DataPreprocessor(lematize = True,
                            stem = False,
                            min_word_len = 3)
        df = dp.preprocess_dataframe(df,
                                    text_column = ["title", "selftext"],
                                    dest_column = "processed_text",
                                    remove_empty_rows = True)
        inputdata = InputData()
        inputdata.texts_from_df(df, column = "processed_text")


    if exists("nmfmodel.obj") and exists("ldamodel.obj"): #files exists 
        with open("nmfmodel.obj", "rb") as f:
            nmfmodel = pickle.load(f)
        with open("ldamodel", "rb") as f:
            ldamodel = pickle.load(f)
    else:
        models = [NMFModel(), LDAModel()]
        output = []

        for model in models:
            model.fit(inputdata)
            output.append(model.get_output())
        
        nmfmodel =  model[0]
        nmfmodel.save("nmfmodel.obj")
        ldamodel = model[1]
        ldamodel.save("ldamodel.obj")

    if exists("scores_df.csv"):
        scores_df = pd.read_csv("scores_df.csv")
    else:
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

        scores = []
        for model in models:
            print("starting next model")
            scores.append([metric.evaluate(inputdata, model.get_output()) for metric in metrics])
        scores_df = pd.DataFrame(scores, index=model_names, columns=metric_names)

    return inputdata, models, scores_df

