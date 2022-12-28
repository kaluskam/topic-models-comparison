import os
import dash
from dash import Dash, Output, Input, State, dcc
import dash_bootstrap_components as dbc

import definitions as d
from app_components.drawer import drawer

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css],
           use_pages=True, pages_folder=os.path.join(d.ROOT_DIR, 'app_components'))

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink('Download subreddits', href='/download-subreddits')),
        dbc.NavItem(
            dbc.NavLink('Modeling', href='/modeling')),
        dbc.NavItem(
            dbc.NavLink('Model comparison', href='/metrics')),
        dbc.NavItem(
            dbc.NavLink('Data Exploration', href='/data-exploration')),
        drawer
    ],
    brand='Topic models comparison',
    brand_href='/',
    color='primary',
    dark=True
)

app.layout = dbc.Container([
    dcc.Store(id='store-multiselect-subreddits', storage_type='local', data=['worldnews']),
    dcc.Store(id='store-date-range', storage_type='local', data=[d.START_DATE, d.END_DATE]),
    navbar,
    dash.page_container
],
    fluid=True
)


@app.callback(Output('store-multiselect-subreddits', 'data'),
              [Input('subreddits-multiselect', 'value')],
              [State('store-multiselect-subreddits', 'data')])
def update_store(value, data):
    return value


@app.callback(Output('multiselect-subreddits', 'value'),
              [Input('store-multiselect-subreddits', 'value')],
              [State('store-multiselect-subreddits', 'data')])
def update_exploration_page(value, data):
    return data


@app.callback(Output('store-date-range', 'data'),
              [Input('date-range-picker', 'value')],
              [State('store-date-range', 'data')])
def update_store_date(value, data):
    return value


@app.callback(Output('date-range', 'value'),
              [Input('store-date-range', 'value')],
              [State('store-date-range', 'data')])
def update_exploration_page_date(value, data):
    return data


if __name__ == '__main__':
    app.run_server(debug=False)
