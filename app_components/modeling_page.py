import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, callback

import pandas as pd
import datetime as dt

import definitions as d
from utils.dashboard_utils import get_data_for_subreddit_select
from utils.visualizer import visualise_topics_overtime, generate_wordcloud, \
    Visualizer
from utils.data_loading import load_downloaded_data, check_cache_existance, create_output_data_cache_filepath, load_cache_output_data
from models.nmf_model import NMFModel
from models.lda_model import LDAModel
from models.berttopic_model import BERTopicModel


dash.register_page(__name__, path='/modeling')

MODEL_NAMES_DICT = {
    'nmf': NMFModel(),
    'lda': LDAModel(),
    'bertopic': BERTopicModel()
}

default_data = load_downloaded_data(['recipes'],
                         [d.END_DATE - dt.timedelta(days=364), d.END_DATE])
default_wordcloud = generate_wordcloud(default_data)

vs = Visualizer()
## MACRO
model = NMFModel()
model.fit(default_data, 5)
output = model.get_output()
texts_topics_df = output.texts_topics
r = pd.merge(default_data.df, texts_topics_df, left_index=True,
             right_on='text_id')
default_overtime_graph_macro = visualise_topics_overtime(r, 'date', output,
                                                         'Topics over time macro',
                                                         'month')
default_topic_similarity_macro = vs.visualize_topics_in_documents(default_data,
                                                                  output,
                                                                  'Topics similarity macro')

##MICRO
model = NMFModel()
model.fit(default_data, 20)
output = model.get_output()
texts_topics_df = output.texts_topics
r = pd.merge(default_data.df, texts_topics_df, left_index=True,
             right_on='text_id')
default_overtime_graph_micro = visualise_topics_overtime(r, 'date', output,
                                                         'Topics over time micro',
                                                         'month')
default_topic_similarity_micro = vs.visualize_topics_in_documents(default_data,
                                                                  output,
                                                                  'Topics similarity micro')

wordcloud_graph = dcc.Graph(id='wordcloud-graph', figure=default_wordcloud,
                          config={"staticPlot": True})


subreddit_select = html.Div([
    dmc.MultiSelect(
        id='subreddits-multiselect',
        label='Subreddits',
        data=get_data_for_subreddit_select(),
        value=['recipes']
    )]
)

model_select = html.Div([
    dmc.Select(
        id='topic-model-select',
        label='Topic model',
        data=[
            {'label': 'NMF', 'value': 'nmf'},
            {'label': 'LDA', 'value': 'lda'},
            {'label': 'BERTopic', 'value': 'bertopic'}],
        value='nmf'
    )]
)

n_topics_input_macro = html.Div([
    dmc.TextInput(id='n-topics-input-macro',
                  label='Number of topics for macro analysis',
                  type='number',
                  value=5)
])

n_topics_input_micro = html.Div([
    dmc.TextInput(id='n-topics-input-micro',
                  label='Number of topics for micro analysis',
                  type='number',
                  description='Type larger number than in macro analysis.\n '
                              'This will give you insights into more specific topics.',
                  value=20)
])

time_interval_radio_buttons = html.Div([
    dmc.RadioGroup(id='time-interval-radios',
                   label='Time interval unit',
                   data=[{'label': 'Day', 'value': 'day'},
                         {'label': 'Week', 'value': 'week'},
                         {'label': 'Month', 'value': 'month'},
                         {'label': 'Year', 'value': 'year'}],
                   value='month',
                   size='sm')])

date_range_picker = html.Div([
    dmc.DateRangePicker(id='date-range-picker',
                        label='Date range',
                        minDate=d.START_DATE,
                        maxDate=d.END_DATE,
                        value=[d.END_DATE - dt.timedelta(days=364), d.END_DATE])])

run_analysis_button = html.Div([
    dmc.Button('Run analysis',
               id='run-analysis-button',
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
        model_select
    ]),
    dmc.Space(h=10),
    html.Div([
        n_topics_input_macro
    ]),
    dmc.Space(h=10),
    html.Div([
        n_topics_input_micro
    ]),
    dmc.Space(h=10),
    html.Div([
        time_interval_radio_buttons
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

layout = dbc.Container([
    dmc.Space(h=20),
    dmc.Alert(id='alert-modeling',
              title=d.ALERT_TITLE,
              children=d.ALERT_MESSAGE,
              color='red'),
    dmc.Space(h=20),
    dbc.Row([
        dbc.Col(controls_card, md=4),
        dbc.Col(wordcloud_graph, md=8)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='topic-similarity-macro',
                          figure=default_topic_similarity_macro), md=6),
        dbc.Col(dcc.Graph(id='topic-similarity-micro',
                          figure=default_topic_similarity_micro), md=6)
    ]),
    dbc.Row([
        dcc.Graph(id='topics-over-time-graph-macro',
                  figure=default_overtime_graph_macro)
    ]),
    dbc.Row([
        dcc.Graph(id='topics-over-time-graph-micro',
                  figure=default_overtime_graph_micro)
    ])
],
    fluid=True
)


@callback(
    Output(component_id='run-analysis-button', component_property='n_clicks'),
    Output(component_id='wordcloud-graph', component_property='figure'),
    Output(component_id='topics-over-time-graph-macro',
           component_property='figure'),
    Output(component_id='topics-over-time-graph-micro',
           component_property='figure'),
    Output(component_id='topic-similarity-macro', component_property='figure'),
    Output(component_id='topic-similarity-micro', component_property='figure'),
    Output(component_id='drawer-subreddits', component_property='value'),
    Output(component_id='drawer-topic-model-select', component_property='value'),
    Output(component_id='drawer-n-topics-input-macro', component_property='value'),
    Output(component_id='drawer-date-range-picker', component_property='value'),
    Output(component_id='alert-modeling', component_property='hide'),
    Input(component_id='run-analysis-button', component_property='n_clicks'),
    Input(component_id='subreddits-multiselect', component_property='value'),
    Input(component_id='topic-model-select', component_property='value'),
    Input(component_id='n-topics-input-macro', component_property='value'),
    Input(component_id='n-topics-input-micro', component_property='value'),
    Input(component_id='time-interval-radios', component_property='value'),
    Input(component_id='date-range-picker', component_property='value'))
def run_analysis(n_clicks, subreddits, topic_model, n_topics_macro,
                 n_topics_micro,
                 time_interval_unit, date_range):

    if n_clicks is not None and n_clicks >= 1:
        input_data = load_downloaded_data(subreddits, date_range)

        wc = generate_wordcloud(input_data)

        model = MODEL_NAMES_DICT[topic_model]

        ## MACRO
        if len(subreddits) == 1:
            n_topics_macro_to_pass = str(n_topics_macro)
            cache_exists = check_cache_existance(subreddits[0], date_range, topic_model, n_topics_macro_to_pass)
            if cache_exists:
                print("Macro was cached")
                cache_file = create_output_data_cache_filepath(subreddits[0], date_range, topic_model, n_topics_macro_to_pass)
                output = load_cache_output_data(cache_file)
            else:
                print("Macro was not cached")
                print(f"This path was not found {create_output_data_cache_filepath(subreddits[0], date_range, topic_model, n_topics_macro_to_pass)}")
                model_topics = int(n_topics_macro) if n_topics_macro is not None else 10
                model.fit(input_data, model_topics)
                output = model.get_output()
        else:
                print("Macro was not cached")
                model_topics = int(n_topics_macro) if n_topics_macro is not None else 10
                model.fit(input_data, model_topics)
                output = model.get_output()


        texts_topics_df = output.texts_topics
        r = pd.merge(input_data.df, texts_topics_df, left_index=True,
                     right_on='text_id')
        fig_macro = visualise_topics_overtime(r, 'date', output,
                                              'Topics over time macro',
                                              time_interval_unit)
        fig_topic_similarity_macro = vs.visualize_topics_in_documents(
            input_data, output, 'Topics similarity macro')

        ## MICRO
        if len(subreddits) == 1:
            n_topics_micro_to_pass = str(n_topics_micro)
            cache_exists = check_cache_existance(subreddits[0], date_range, topic_model, n_topics_micro_to_pass)
            if cache_exists:
                print("Micro was cached")
                cache_file = create_output_data_cache_filepath(subreddits[0], date_range, topic_model, n_topics_micro_to_pass)
                output = load_cache_output_data(cache_file)
            else:
                print("Micro was not cached")
                print(f"This path was not found {create_output_data_cache_filepath(subreddits[0], date_range, topic_model, n_topics_micro_to_pass)}")
                model_topics = int(n_topics_micro)
                model.fit(input_data, model_topics)
                output = model.get_output()
        else:
            print("Micro was not cached")
            model_topics = int(n_topics_micro)
            model.fit(input_data, model_topics)
            output = model.get_output()

        texts_topics_df = output.texts_topics
        r = pd.merge(input_data.df, texts_topics_df, left_index=True,
                     right_on='text_id')
        fig_micro = visualise_topics_overtime(r, 'date', output,
                                              'Topics over time micro',
                                              time_interval_unit)
        fig_topic_similarity_micro = vs.visualize_topics_in_documents(
            input_data, output, 'Topics similarity micro')

        return 0, wc, fig_macro, fig_micro, fig_topic_similarity_macro, \
               fig_topic_similarity_micro, subreddits, topic_model, \
               n_topics_macro, date_range, True

    else:
        raise Exception("Doesn't change anything")


