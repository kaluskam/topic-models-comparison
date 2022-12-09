import datetime as dt

import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash import html, Output, Input, callback

import definitions as d
from utils.dashboard_utils import get_data_for_subreddit_select

subreddit_select = html.Div([
    dmc.MultiSelect(
        id='drawer-subreddits',
        class_name='drawer-disabled',
        label='Subreddits',
        data=get_data_for_subreddit_select(),
        value=['recipes'],
        disabled=True
    )]
)
model_select = html.Div([
    dmc.Select(
        id='drawer-topic-model-select',
        class_name='drawer-disabled',
        label='Topic model',
        data=[
            {'label': 'NMF', 'value': 'nmf'},
            {'label': 'LDA', 'value': 'lda'},
            {'label': 'BERTopic', 'value': 'bertopic'}],
        value='nmf',
        disabled=True
    )]
)
n_topics_input_macro = html.Div([
    dmc.TextInput(id='drawer-n-topics-input-macro',
                  class_name='drawer-disabled',
                  label='Number of topics for macro analysis',
                  type='number',
                  value=5,
                  disabled=True)
])
date_range_picker = html.Div([
    dmc.DateRangePicker(id='drawer-date-range-picker',
                        label='Date range',
                        class_name='drawer-disabled',
                        minDate=d.START_DATE,
                        maxDate=d.END_DATE,
                        value=[d.END_DATE - dt.timedelta(days=365), d.END_DATE],
                        disabled=True)])

options_card =dbc.Card([
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
        date_range_picker
    ])
],
    body=True)

drawer = html.Div(
    [
        dmc.Button("Show chosen options", id="drawer-button",
                   style={'background-color': '#0d6efd', 'padding-top': '5px'}),
        dmc.Drawer(
            children=[options_card],
            title="Analysis options",
            id="drawer",
            padding="md",
            size="30%"
        ),
    ]
)


@callback(
    Output("drawer", "opened"),
    Input("drawer-button", "n_clicks"),
    prevent_initial_call=True,
)
def drawer_demo(n_clicks):
    return True

