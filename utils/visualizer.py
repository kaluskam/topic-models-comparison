import plotly.express as px
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP


class Visualizer:
    def __init__(self, parameters=None):
        self.parameters = parameters
        if parameters is None:
            self.init_default_parameters()

        pass

    def visualize_words_in_topic(self, topic):
        topics_df = pd.DataFrame(
            {"words": topic.words, "scores": topic.word_scores})
        fig = px.bar(topics_df, x="scores", y="words")
        fig.update_layout(yaxis=dict(autorange="reversed"))

        return fig

    def visualize_topics_in_documents(self, inputData, outputData):
        self.tfidf = TfidfVectorizer(**self.parameters["tfidf"])
        self.tfidf.fit(inputData.texts)
        topics = [topic.words for topic in outputData.topics]
        topic_names = [i for i in range(outputData.n_topics)]
        frequencies = [topic.frequency for topic in outputData.topics]
        texts = [' '.join(topic.words[0:5]) for topic in outputData.topics]
        embeddings = self.tfidf.transform(topics).toarray()
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = UMAP(**self.parameters["umap"]).fit_transform(embeddings)

        df = pd.DataFrame({"x": embeddings[:, 0],
                           "y": embeddings[:, 1],
                           "Texts": texts,
                           "Topic": topic_names,
                           "Size": frequencies})

        fig = px.scatter(df,
                         x="x",
                         y="y",
                         size="Size",
                         size_max=40,
                         template="simple_white",
                         labels={"x": "", "y": ""},
                         hover_data={"Topic": True,
                                     "Texts": True,
                                     "Size": False,
                                     "x": False,
                                     "y": False})
        return fig

    def init_default_parameters(self):
        self.parameters = {"tfidf": {'preprocessor': ' '.join},
                           "umap": {"n_neighbors": 2,
                                    "n_components": 2,
                                    "metric": "hellinger",
                                    "random_state": 123}}


def visualise_topics_overtime(df, date_column, outputdata, interval='month'):
    assert interval in ('day', 'week', 'month', 'year')
    if interval == 'month':
        df['month'] = df[date_column].apply(
            lambda date: dt.datetime.strptime(date, "%Y-%m-%d").month)
        df['year'] = df[date_column].apply(
            lambda date: dt.datetime.strptime(date, "%Y-%m-%d").year)
        df_vis = df.groupby(['year', 'month', 'topic_id']).size().reset_index(
            name='counts')
        df_vis['date'] = df_vis.apply(lambda row: f'{row.year}-{row.month}-01',
                                      axis=1)
    elif interval == 'year':
        # TODO
        pass
    elif interval == 'week':
        # TODO
        pass

    df_vis['topic'] = df_vis['topic_id'].apply(lambda topic_id: ' '.join(
        outputdata.topics_dict[topic_id].get_words()))

    fig = px.line(df_vis, x='date', y='counts', color='topic', markers=True,
                  labels={
                      'topic_id': 'Topic number',
                      'topic': 'Topic'
                  },
                  hover_data={'topic': True})
    fig.update_traces(marker={'size': 12})
    # opcjonalne formatowanie daty
    # fig.update_xaxes(
    #     dtick='M1',
    #     tickformat='%b\n%Y'
    # )
    fig.show()
