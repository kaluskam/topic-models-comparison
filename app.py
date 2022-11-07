import streamlit as st

def create_metrics():
    umass = UMassCoherenceMetric()
    cv = CVCoherenceMetric()
    cuci = CUCICoherenceMetric()
    cnpmi = CNPMICoherenceMetric()
    return [umass, cv, cuci, cnpmi]

@st.cache
def load_data():
    df = pd.read_csv("data/aita_df.csv")
    dp = DataPreprocessor(True, False)
    text_df = dp.preprocess_dataframe(df.loc[1:500, ], "selftext", "processed_text", True)
    datamodel = InputData()
    datamodel.texts_from_df(text_df, "processed_text")
    return datamodel

@st.cache
def create_nmf_topics(datamodel):
    nmf = NMFModel()
    nmf.fit(datamodel)
    nmf_topics = nmf.get_topics()
    return nmf_topics

@st.cache
def create_lda_topics(datamodel):
    lda = LDAModel()
    lda.fit(datamodel)
    lda_topics = lda.get_topics()
    return lda_topics

@st.cache
def create_bert_topics(datamodel):
    bert = BERTopicModel()
    bert.fit(datamodel)
    bert_topics = bert.get_topics()
    return bert_topics

@st.cache
def calculate_metrics(datamodel, metrics, topics):
    return [[metric.evaluate(datamodel, topics[0]) for metric in metrics],
            [metric.evaluate(datamodel, topics[1]) for metric in metrics],
            [metric.evaluate(datamodel, topics[2]) for metric in metrics]]

@st.cache
def create_topics(datamodel):
    return [create_nmf_topics(datamodel), create_lda_topics(datamodel), create_bert_topics(datamodel)]

if __name__ == "__main__":
    import streamlit as st

    from utils.data_structures import InputData
    from models.nmf_model import NMFModel
    from models.lda_model import LDAModel
    from models.berttopic_model import BERTopicModel
    from utils.preprocessing import DataPreprocessor 
    from metrics.coherence_metric import CNPMICoherenceMetric, CUCICoherenceMetric, CVCoherenceMetric, UMassCoherenceMetric
    from utils.visualizer import Visualizer
    from streamlit_plotly_events import plotly_events

    import pandas as pd

    metrics = create_metrics()
    datamodel = load_data()
    topics = create_topics(datamodel)

    results = pd.DataFrame(calculate_metrics(datamodel, metrics, topics),
        columns = ["umass", "cv", "cuci", "cnpmi"],
        index = ["NMF", "LDA", "BERTopic"])

    vs = Visualizer()


    with st.container():
        st.write(f"MMF")
        selected_topic = plotly_events(vs.visualize_topics_in_documents(datamodel, topics[0]))
        if selected_topic:
            st.plotly_chart(vs.visualize_words_in_topic(topics[0].topics[selected_topic[0]["pointIndex"]]))

    with st.container():
        st.write(f"LDA")
        selected_topic = plotly_events(vs.visualize_topics_in_documents(datamodel, topics[1]))
        if selected_topic:
            st.plotly_chart(vs.visualize_words_in_topic(topics[1].topics[selected_topic[0]["pointIndex"]]))
            
    st.table(results)

