import plotly.express as px
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
from wordcloud import WordCloud
import matplotlib.pyplot as plt

FONT = dict(
            size=14,
            color='Black'
        )

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

    def visualize_topics_in_documents(self, inputData, outputData, title):
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
                                     "y": False},
                         title=title)
        fig.update_layout(font=FONT)
        return fig

    def init_default_parameters(self):
        self.parameters = {"tfidf": {'preprocessor': ' '.join},
                           "umap": {"n_neighbors": 2,
                                    "n_components": 2,
                                    "metric": "hellinger",
                                    "random_state": 123}}


def visualise_topics_overtime(df, date_column, outputdata, title, interval='month'):
    xaxis_title = ''
    ticktext = []
    tickformat = None
    if interval == 'year':
        df['year'] = df[date_column].apply(
            lambda date: dt.datetime.strptime(date, "%Y-%m-%d").year)
        df_vis = df.groupby(['year', 'topic_id']).size().reset_index(
            name='counts')
        df_vis['x_axis_date'] = df_vis['year']
        xaxis_title = 'Year'
        ticktext = sorted(df['year'].unique())

    elif interval == 'month':
        df['month'] = df[date_column].apply(
            lambda date: dt.datetime.strptime(date, "%Y-%m-%d").month)
        df['year'] = df[date_column].apply(
            lambda date: dt.datetime.strptime(date, "%Y-%m-%d").year)
        df_vis = df.groupby(['year', 'month', 'topic_id']).size().reset_index(
            name='counts')
        df_vis['x_axis_date'] = df_vis.apply(lambda row: f'{row.year}-{row.month}',
                                      axis=1)
        xaxis_title = 'Month'
        ticktext = df_vis['x_axis_date']
        tickformat = '%b, %Y'

    elif interval == 'week':
        df['week'] = df[date_column].apply(
            lambda date: dt.datetime.strptime(date, "%Y-%m-%d").isocalendar().week)
        df['year'] = df[date_column].apply(
            lambda date: dt.datetime.strptime(date, "%Y-%m-%d").year)
        df_vis = df.groupby(['year', 'week', 'topic_id']).size().reset_index(
            name='counts')
        df_vis['x_axis_date'] = df_vis.apply(
            lambda row: f'Week {row.week}, {row.year}',
            axis=1)
        xaxis_title = 'Week'
        ticktext = df_vis['x_axis_date']

    elif interval == 'day':
        df_vis = df.groupby([date_column, 'topic_id']).size().reset_index(
            name='counts')
        df_vis['x_axis_date'] = df_vis[date_column]
        xaxis_title = 'Day'
        ticktext = df_vis['x_axis_date']

    df_vis['topic'] = df_vis['topic_id'].apply(lambda topic_id: ' '.join(
        outputdata.topics[topic_id].get_words()))

    fig = px.line(df_vis, x='x_axis_date', y='counts', color='topic', markers=True,
                  labels={
                      'topic_id': 'Topic number',
                      'topic': 'Topic',
                      'x_axis_date': xaxis_title,
                      'counts': 'Counts'
                  },
                  hover_data={'topic': True})
    fig.update_traces(marker={'size': 12})
    fig.update_layout(
        plot_bgcolor='rgba(237, 250, 253, 0.5)',
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title='Counts',
        xaxis={
            'tickvals': ticktext,
            'tickformat': tickformat
        },
        font=FONT)

    return fig


def generate_wordcloud(input_data):
    words = ''
    for text in input_data.texts:
        words += ' '.join(text) + ' '

    word_cloud = WordCloud(background_color='white', min_font_size=14, width=1000, height=500).generate(words)
    wc = px.imshow(word_cloud.to_image())
    wc.update_xaxes(visible=False)
    wc.update_yaxes(visible=False)
    return wc

