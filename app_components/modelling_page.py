import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
import plotly.express as px
import datetime as dt

from utils.visualizer import visualise_topics_overtime, generate_wordcloud
from models.nmf_model import NMFModel
from utils.data_structures import InputData
START_DATE = '2020-10-01'

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

df_wc = pd.read_csv('data\\preprocessed\\askmen.csv')
input_wc = InputData()
input_wc.texts_from_df(df_wc, 'lematized')
nmf = NMFModel()
nmf.fit(input_wc)
out = nmf.get_output()
print(nmf.output.texts_topics)
texts_topics_df = nmf.output.texts_topics


r = pd.merge(df_wc, texts_topics_df, left_index=True, right_on='text_id')
fig = visualise_topics_overtime(r, 'date', nmf.output)



## Components

subreddit_select = html.Div([
    dbc.Select(
    id='subreddit-select',
    options=[
        {'label': 'AskMen', 'value': 'askmen'},
        {'label': 'AskWomen', 'value': 'askwomen'},
        {'label': 'AskReddit', 'value': 'askreddit'}]
    )]
)

model_select = html.Div([
    dbc.Select(
    id='model-select',
    options=[
        {'label': 'NMF', 'value': 'nmf'},
        {'label': 'LDA', 'value': 'lda'},
        {'label': 'BertTopic', 'value': 'bertopic'}]
    )]
)

time_interval_radio_buttons = dbc.RadioItems(id='time-interval-radios',
                   options=[{'label': 'Day', 'value': 0},
                            {'label': 'Week', 'value': 1},
                            {'label': 'Month', 'value': 2}],
                   # className='btn-group',
                   # inputClassName="btn-check",
                   # labelClassName="btn btn-outline-primary",
                   # labelCheckedClassName="active",
                   value=1)

time_interval_range_slider = dcc.RangeSlider(min=0, max=10, step=1,
                                             id='time-range-slider')

controls_card = dbc.Card([
    html.Div([
        dbc.Label('Subreddit'),
        subreddit_select
    ]),
    html.Div([
        dbc.Label('Topic model'),
        model_select
    ]),
    html.Div([
        dbc.Label('Time interval'),
        time_interval_radio_buttons
    ]),
    html.Div([
        dbc.Label('Time range'),
        time_interval_range_slider
    ])
],
    body=True

)
# user_panel = dbc.Container(
#     children=[
#         dbc.Row(children=[
#             dbc.Col(
#                 children=[subreddit_select],
#                 width={'size': 4}
#             ),
#             dbc.Col(
#                 children=[
#                     dcc.Graph(figure=fig)
#                 ],
#                 width={'size': 8}
#             )]),
#         dbc.Row(children=[
#             dbc.Col(children=[model_dropdown])
#         ])
#     ]
# )

modelling_page = dbc.Container([
    dbc.Row([
        dbc.Col(controls_card, md=4),
        dbc.Col(dcc.Graph(id='cluster-graph', figure=px.imshow(generate_wordcloud(input_wc))), md=8)
    ],
    align='center'),
    dbc.Row([
        dcc.Graph(figure=fig)
    ])
    ],
    fluid=True
)
