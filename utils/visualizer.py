import numpy as np
import plotly.express as px
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from umap import UMAP
from wordcloud import WordCloud
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import numpy
import os
import definitions as d
numpy.set_printoptions(threshold=sys.maxsize)

FONT = dict(
            size=14,
            color='Black'
        )

class Visualizer:
    """
   Class which wraps up together methods for creating the visualizations
   """
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
        """
        Method to visualize topic clusters using UMAP. Can be used to choose the optimal number of topic.

        Parameters
        ----------
        inputData : InputData
            input data of a model
        outputData: OutputData
            output data of a model
        title: str
            title of a plot

        Returns
        -------
        fig: plotly.graph_objs._figure.Figure
            scatter plot
        """
        self.tfidf = TfidfVectorizer(**self.parameters["tfidf"])
        self.tfidf.fit(inputData.texts)
        topics = [topic.words for topic in outputData.topics]
        topic_names = [i for i in range(outputData.n_topics)]
        frequencies = [topic.frequency for topic in outputData.topics]
        max_freq = max(frequencies)/100
        frequencies = list(map(lambda x: x if x>max_freq else max_freq, frequencies)) #make all circles visible
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
                         labels={"x": "UMAP component 1", "y": "UMAP component 2"},
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
                                    "min_dist": 0.99,
                                    "metric": "hellinger",
                                    "random_state": 123}}


def visualise_topics_overtime(df, date_column, outputdata, title, interval='month', google_trends_df=None):
    """
        Method to visualize topics over time with a selected granularity.

        Parameters
        ----------
        df : DataFrame
            reference dataframe
        date_column: str
            column containing posts creation date
        outputdata: OutputData
            output data of a model
        title: str
            title of a plot
        interval: str
            granularity of topics over time visualization

        Returns
        -------
        fig: plotly.graph_objs._figure.Figure
            line chart
        """
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
        outputdata.topics[topic_id].get_words()) if len(outputdata.topics) > topic_id and topic_id >= 0 else "")

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
    """
    Create a wordcloud using words from InputData object
    """
    words = ''
    for text in input_data.texts:
        words += ' '.join(text) + ' '

    word_cloud = WordCloud(background_color='white', min_font_size=14, width=1400, height=600).generate(words)
    wc = px.imshow(word_cloud.to_image())
    wc.update_traces(hovertemplate=None, hoverinfo="skip")
    wc.update_xaxes(visible=False)
    wc.update_yaxes(visible=False)
    return wc

def plot_popular_words(input_data):
    """
    Plot top 10 most popular words for a given InputData object instance
    """
    words = ''
    for text in input_data.texts:
        words += ' '.join(text) + ' '
    word_list = words.replace(']','').replace('[', '').split(sep=' ')
    word_list = [str(word).replace("'","") for word in word_list]
    counted_words = Counter(word_list).most_common(10)
    df = pd.DataFrame(counted_words, columns=['word', 'count']).sort_values('count', ascending=True)
    fig = px.bar(df, x="count", y="word", orientation='h', title="Popular words")
    fig.update_layout(
        plot_bgcolor='rgba(237, 250, 253, 0.5)',
        font=FONT)
    return fig


def plot_posts_distribution(dfs, date_range, aggregation_level='default'):
    """
    Plot number of posts for a selected granularity level. Can be used for checking data representativeness over a selected period.
    """
    i = 0
    days_interval = pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])
    days_interval = int(days_interval.days)
    if aggregation_level == 'default':
        if days_interval <= 60:
            aggregation_level = 'daily'
        elif days_interval <= 365:
            aggregation_level = 'weekly'
        else:
            aggregation_level = 'monthly'
    subreddits = pd.unique(dfs.subreddit)
    for subreddit in subreddits:
        df_counted = dfs.loc[dfs['subreddit'] == subreddit].date.value_counts().reset_index()
        df_counted.columns = ['date', 'posts number']
        df_counted = df_counted.sort_values('date').reset_index(drop=True)
        if aggregation_level == 'weekly':
            dates = df_counted[df_counted.index // 7 == df_counted.index / 7].date.reset_index(drop=True)
            df_counted= df_counted.groupby(df_counted.index // 7).sum() #weekly aggreagation
            df_counted['date'] = dates
        elif aggregation_level == 'monthly':
            df_counted.date = pd.to_datetime(df_counted.date.apply(lambda x: x.strftime('%m/%Y')))
            df_counted = df_counted.groupby('date').sum().reset_index()
        df_counted['subreddit'] = subreddit
        if i == 0:
            fig = px.line(df_counted, x='date', y='posts number', title=f"{aggregation_level.capitalize()} number of posts on subreddit(s)", color='subreddit')
        else:
            fig.add_scatter(x=df_counted['date'], y=df_counted['posts number'], mode='lines', name=subreddit)
        i += 1
        fig.update_layout(
            plot_bgcolor='rgba(237, 250, 253, 0.5)',
            font=FONT)
    return fig


def get_ngrams(df, column='raw_text', ngram_range=(2,2)):
    """
    Retrieve the ngrams (bigrams by default). Part of a data exploration analysis.
    """
    count_vectorizer = CountVectorizer(min_df = 10, stop_words='english', token_pattern = r"[a-zA-Z]{2,}", ngram_range=ngram_range)
    count_cat = count_vectorizer.fit_transform(df[column])
    count_feature_names = count_vectorizer.get_feature_names()
    df_count = pd.DataFrame(count_cat.toarray(), columns=list(count_feature_names))
    df_count = df_count.sum(axis=0)
    return dict(sorted(df_count.items(), key=lambda item: item[1], reverse=True))


def plot_wordcloud(df, column='raw_text', ngram_range=(2, 2)):
    """
    Plot word cloud for ngrams (bigrams by default). Part of a data exploration analysis.
    """
    word_counters = get_ngrams(df, column, ngram_range)
    wc = WordCloud(background_color='white', min_font_size=14, width=1000, height=500)
    word_cloud = wc.generate_from_frequencies(frequencies=word_counters)
    wc = px.imshow(word_cloud.to_image())
    wc.update_traces(hovertemplate=None, hoverinfo="skip")
    wc.update_xaxes(visible=False)
    wc.update_yaxes(visible=False)
    return wc


def plot_popular_words_stacked(df):
    """
    Plot popular words with respect to their subreddit origin.
    """
    subreddits = pd.unique(df.subreddit)
    first = True
    for subreddit in subreddits:
        ngrams = get_ngrams(df.loc[df['subreddit']==subreddit, :], 'raw_text', ngram_range=(2, 2))
        res = pd.DataFrame(ngrams, index = [0]).T.reset_index()
        res.columns = ['word', str(subreddit)]
        if first:
            result = res
            first = False
        else:
            result = result.merge(res, how='outer')
    result['count'] = result.loc[:,result.columns != 'word'].sum(axis=1)
    result = result.sort_values('count', ascending=False).iloc[0:10,:].sort_values('count')
    fig = px.bar(result, y="word", x=subreddits, title="Popular bigrams", orientation='h')
    fig.update_layout(
        plot_bgcolor='rgba(237, 250, 253, 0.5)',
        font=FONT)
    return fig


def plot_word_count_distribution(df, column='raw_text'):
    """
    Plot distribution of number of words inside a post. Part of a data exploration analysis.
    """
    a = df[column].apply(lambda x: len(str(x).split(' ')))
    a = a.sort_values()[0:int(len(a)*0.98)] #delete 1% of extreme cases
    a.columns = ['number of posts']
    fig = px.histogram(a, title='Posts word count distribution')
    fig.update_layout(
        plot_bgcolor='rgba(237, 250, 253, 0.5)',
        font=FONT,
        showlegend=False)
    return fig


def plot_topic_overtime_vs_google_trends(output_data, topic_id, input_data, google_trends_df, fig_title, filename):
    texts_topics_df = output_data.texts_topics
    df = pd.merge(input_data.df, texts_topics_df, left_index=True, right_on='text_id')
    df = df[df['topic_id'] == topic_id]
    df['month'] = df['date'].apply(
        lambda date: dt.datetime.strptime(date, "%Y-%m-%d").month)
    df['year'] = df['date'].apply(
        lambda date: dt.datetime.strptime(date, "%Y-%m-%d").year)
    df_vis = df.groupby(['year', 'month', 'topic_id']).size().reset_index(
        name='counts')
    df_vis['x_axis_date'] = df_vis.apply(
        lambda row: f'{row.year}-{row.month if row.month >= 10 else "0" + str(row.month)}-01', axis=1)
    x_axis_data = df_vis['x_axis_date']
    y1_axis_data = df_vis['counts']

    google_trends_df['month'] = google_trends_df['date'].apply(
        lambda date: dt.datetime.strptime(date, "%Y-%m-%d").month)
    google_trends_df['year'] = google_trends_df['date'].apply(
        lambda date: dt.datetime.strptime(date, "%Y-%m-%d").year)
    google_trends_df['x_axis_date'] = google_trends_df.apply(
        lambda row: f'{row.year}-{row.month if row.month >= 10 else "0" + str(row.month)}-01', axis=1)
    google_trends_df['counts'] = google_trends_df['counts'].replace('<1', 0).astype(int)
    y2_axis_data = google_trends_df[['x_axis_date', 'counts']].groupby(['x_axis_date']).sum()
    print(x_axis_data)
    x_axis_data = x_axis_data.apply(lambda x: np.datetime64(x))

    fig, ax1 = plt.subplots()
    fig.suptitle(fig_title, size=16)
    ax1.set_title('Topic over time vs. Google Trends')
    color1 = '#219EBC'
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of posts', color=color1)
    ax1.plot(x_axis_data, y1_axis_data, color=color1, linewidth=3)
    ax1.tick_params(axis='y')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))

    ax2 = plt.twinx(ax1)
    color2 = '#E63946'
    ax2.set_ylabel('Google trends score', color=color2)
    ax2.plot(x_axis_data, y2_axis_data, color=color2, linewidth=3)
    ax2.tick_params(axis='y')

    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    fig.tight_layout()
    plt.savefig(os.path.join(d.EXPERIMENTS_DIR, 'images', filename), dpi=300)


def plot_topics_overtime_comparison(dataframe, date_column, outputdata, indices, title, filename, interval='month',
                            models=['LDA', 'NMF', 'BERTopic'], default_legend=True):
    fig, ax = plt.subplots(1, figsize=(9, 4.8))
    colors = ['#219EBC', '#E63946', 'green']

    for i in range(len(indices)):

        df = pd.merge(dataframe.df, outputdata[i].texts_topics, left_index=True,
                      right_on='text_id')
        xaxis_title = ''
        if interval == 'year':
            df['year'] = df[date_column].apply(
                lambda date: dt.datetime.strptime(date, "%Y-%m-%d").year)
            df_vis = df.groupby(['year', 'topic_id']).size().reset_index(
                name='counts')
            df_vis['x_axis_date'] = df_vis['year']
            xaxis_title = 'Year'


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
            df_vis = df_vis.loc[df_vis.topic_id == indices[i], :]
            df_vis['x_axis_date'] = pd.to_datetime(df_vis['x_axis_date'])

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
            df_vis['x_axis_date'] = pd.to_datetime(df_vis.week.astype(str) +
                                                   df_vis.year.astype(str).add('-1'), format='%V%G-%u')

        elif interval == 'day':
            df_vis = df.groupby([date_column, 'topic_id']).size().reset_index(
                name='counts')
            df_vis['x_axis_date'] = df_vis[date_column]
            xaxis_title = 'Day'
            df_vis['x_axis_date'] = pd.to_datetime(df_vis['x_axis_date'])

        df_vis['topic'] = df_vis['topic_id'].apply(lambda topic_id: ' '.join(
            outputdata[i].topics[topic_id].get_words()) if len(
            outputdata[i].topics) > topic_id and topic_id >= 0 else "")
        df_vis['model'] = models[i]
        df_vis = df_vis.loc[df_vis.topic_id == indices[i], :]
        if default_legend:
            plt.plot(df_vis['x_axis_date'], df_vis['counts'], label=models[i], color=colors[i])
            plt.legend(loc='upper right')
        else:
            plt.plot(df_vis['x_axis_date'], df_vis['counts'], label=models[i] + " keywords: " + df_vis['topic'].iloc[0],
                     color=colors[i])
            plt.legend(loc='lower left', bbox_to_anchor=(-0.03, -0.35))
        plt.title(title, size=16)

        plt.ylabel('Count')
        plt.savefig(os.path.join(d.EXPERIMENTS_DIR, 'images', filename), dpi=300)

    return fig