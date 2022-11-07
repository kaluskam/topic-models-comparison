import dash_bootstrap_components as dbc
import dash_core_components as dcc
import datetime as dt
START_DATE = '2020-10-01'

user_panel = dbc.Container(
    children=[
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem('NMF'), dbc.DropdownMenuItem('LDA'), dbc.DropdownMenuItem('BertTopic')],
            label='Model',
            group=True
        ),
        dbc.RadioItems(['Day', 'Week', 'Month'],
                       label='Time interval',
                       value='Month'),
        dcc.RangeSlider(START_DATE, dt.date.today(), id='time-range-slider')
    ]
)

modelling_page = dbc.Container(
    children=[
        user_panel
    ]
)

