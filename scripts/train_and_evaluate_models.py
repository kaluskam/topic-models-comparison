def create_metrics():
    umass = UMassCoherenceMetric()
    cv = CVCoherenceMetric()
    cuci = CUCICoherenceMetric()
    cnpmi = CNPMICoherenceMetric()
    return [umass, cv, cuci, cnpmi]


def load_data():
    df = pd.read_csv("..\\data\\preprocessed\\recipes.csv", sep=';')
    dp = DataPreprocessor(True, False)
    # text_df = dp.preprocess_dataframe(df.loc[1:100, ], "selftext",
    #                                   "processed_text", True)
    datamodel = InputData()
    datamodel.texts_from_df(df, "lematized")
    return datamodel


def create_nmf_topics(datamodel):
    nmf = NMFModel()
    nmf.fit(datamodel)
    nmf_topics = nmf.get_output()
    return nmf_topics


def create_lda_topics(datamodel):
    lda = LDAModel()
    lda.fit(datamodel)
    lda_topics = lda.get_output()
    return lda_topics


def create_bert_topics(datamodel):
    bert = BERTopicModel()
    bert.fit(datamodel)
    bert_topics = bert.get_output()
    return bert_topics


def calculate_metrics(datamodel, metrics, topics):
    return [[metric.evaluate(datamodel, topics[0]) for metric in metrics],
            [metric.evaluate(datamodel, topics[1]) for metric in metrics]]


def create_topics(datamodel):
    return [create_nmf_topics(datamodel), create_lda_topics(datamodel)]


if __name__ == "__main__":
    from utils.data_structures import *
    from models.nmf_model import NMFModel
    from models.lda_model import LDAModel
    from models.berttopic_model import BERTopicModel
    from utils.preprocessing import DataPreprocessor
    from metrics.coherence_metric import *

    import pandas as pd

    metrics = create_metrics()
    datamodel = load_data()
    topics = create_topics(datamodel)

    results = pd.DataFrame(calculate_metrics(datamodel, metrics, topics),
                           columns=["umass", "cv", "cuci", "cnpmi"],
                           index=["NMF", "LDA"])

    print(results)
