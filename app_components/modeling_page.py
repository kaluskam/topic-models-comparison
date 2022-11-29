import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html, Input, Output, callback
import datetime as dt

import definitions as d
from utils.visualizer import visualise_topics_overtime, generate_wordcloud, \
    Visualizer
from utils.data_loading import load_downloaded_data
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
                         [d.END_DATE - dt.timedelta(days=365), d.END_DATE])
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

wordcloud_graph = dcc.Graph(id='wordcloud-graph', figure=default_wordcloud)

layout = dbc.Container([
    dbc.Row([
        dbc.Col(wordcloud_graph, md=12)
    ],
        align='center'),
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
        model.fit(input_data, int(n_topics_macro))
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
        model.fit(input_data, int(n_topics_micro))
        output = model.get_output()
        texts_topics_df = output.texts_topics
        r = pd.merge(input_data.df, texts_topics_df, left_index=True,
                     right_on='text_id')
        fig_micro = visualise_topics_overtime(r, 'date', output,
                                              'Topics over time micro',
                                              time_interval_unit)
        fig_topic_similarity_micro = vs.visualize_topics_in_documents(
            input_data, output, 'Topics similarity micro')
        return 0, wc, fig_macro, fig_micro, fig_topic_similarity_macro, fig_topic_similarity_micro

    else:
        raise Exception("Doesn't change anything")


