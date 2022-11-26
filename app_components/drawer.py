import datetime as dt

import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash import html, Output, Input, callback

import definitions as d
from utils.dashboard_utils import get_data_for_subreddit_select

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
                        value=[d.END_DATE - dt.timedelta(days=365), d.END_DATE])])

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

drawer = html.Div(
    [
        dmc.Button("Open Drawer", id="drawer-demo-button"),
        dmc.Drawer(
            children=[controls_card],
            title="Drawer Example",
            id="drawer",
            padding="md",
        ),
    ]
)


@callback(
    Output("drawer", "opened"),
    Input("drawer-demo-button", "n_clicks"),
    prevent_initial_call=True,
)
def drawer_demo(n_clicks):
    return True