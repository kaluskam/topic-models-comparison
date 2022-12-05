import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, callback
import datetime as dt
from utils.data_loading import load_downloaded_data, load_raw_data
from utils.visualizer import plot_word_count_distribution, generate_wordcloud, \
    Visualizer, plot_popular_words, plot_posts_distribution, plot_wordcloud, plot_popular_words_stacked
import dash
from dash import html
import definitions as d

dash.register_page(__name__, path='/data-exploration')

layout = html.Div(
    html.H1("Strona do eksploracji danych")
)

###
# START_DATE = dt.date(2019, 10, 1)
# END_DATE = dt.date(2022, 9, 30)

default_data = load_downloaded_data(['worldnews'],
                         [d.END_DATE - dt.timedelta(days=365), d.END_DATE])
default_popular_words = plot_popular_words(default_data)

# exploration performed on data before preprocessing
default_raw_data = load_raw_data(['worldnews'],
                         [d.END_DATE - dt.timedelta(days=365), d.END_DATE])
default_wordcloud = plot_wordcloud(default_raw_data, column='raw_text')
default_posts_num_dist = plot_posts_distribution(default_raw_data, [d.END_DATE - dt.timedelta(days=365), d.END_DATE])
default_popular_bigrams = plot_popular_words_stacked(default_raw_data)
default_word_count_hist = plot_word_count_distribution(default_raw_data, column='raw_text')


## Components

subreddit_select = html.Div([
    dmc.MultiSelect(
        id='multiselect-subreddits',
        label='subreddits',
        data=[
            {'label': 'CasualUK', 'value': 'casualuk'},
            {'label': 'WorldNews', 'value': 'worldnews'},
            {'label': 'Science', 'value': 'science'},
            {'label': 'SubredditDrama', 'value': 'subredditdrama'},
            {'label': 'Medical', 'value': 'medical'}],
        value=['worldnews']
    )]
)

date_range_picker = html.Div([
    dmc.DateRangePicker(id='date-range',
                        label='Date range',
                        minDate=d.START_DATE,
                        maxDate=d.END_DATE,
                        value=[d.END_DATE - dt.timedelta(days=365), d.END_DATE])])

run_analysis_button = html.Div([
    dmc.Button('Run analysis',
               id='submit',
               variant='light',
               fullWidth=True,
               )
])

controls_card = dbc.Card([
    html.Div([
        subreddit_select
    ]),
    dmc.Space(h=10),
    html.Div([
        date_range_picker
    ]),
    dmc.Space(h=10),
    html.Div([
        run_analysis_button
    ])
],
    body=True)

wordcloud_graph = dcc.Graph(id='wordcloud', figure=default_wordcloud)
common_words_graph = dcc.Graph(id='popular-words', figure=default_popular_words)
common_bigrams_graph = dcc.Graph(id='popular-bigrams', figure=default_popular_bigrams)
posts_number = dcc.Graph(id='posts-number', figure=default_posts_num_dist)
word_count_hist = dcc.Graph(id='word-count-hist', figure=default_word_count_hist)

# Define the page layout
layout = dbc.Container([
dbc.Row([
        dbc.Col(controls_card, md=4),
        dbc.Col(wordcloud_graph, md=8),
        dbc.Col(common_words_graph, md=6),
        dbc.Col(common_bigrams_graph, md=6),
        dbc.Col(posts_number, md=8),
        dbc.Col(word_count_hist, md=4)
    ],
        align='center'),
    ])


@callback(
    Output(component_id='submit', component_property='n_clicks'),
    Output(component_id='wordcloud', component_property='figure'),
    Output(component_id='popular-words', component_property='figure'),
    Output(component_id='popular-bigrams', component_property='figure'),
    Output(component_id='posts-number', component_property='figure'),
    Output(component_id='word-count-hist', component_property='figure'),
    Input(component_id='submit', component_property='n_clicks'),
    Input(component_id='multiselect-subreddits', component_property='value'),
    Input(component_id='date-range', component_property='value'))
def run_analysis(n_clicks, subreddits, date_range):
    if n_clicks is not None and n_clicks >= 1:

        input_data = load_downloaded_data(subreddits, date_range)
        raw_input_data = load_raw_data(subreddits, date_range)
        wc = plot_wordcloud(raw_input_data)
        popular_words = plot_popular_words(input_data)
        popular_bigrams = plot_popular_words_stacked(raw_input_data)
        posts_num = plot_posts_distribution(raw_input_data, date_range)
        word_count = plot_word_count_distribution(raw_input_data)

        return 0, wc, popular_words, popular_bigrams, posts_num, word_count

    else:
        raise Exception("Doesn't change anything")