import os
import dash
from dash import Dash
import dash_bootstrap_components as dbc

import definitions as d

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
            dbc.NavLink('Data Exploration', href='/data-exploration'))
        ],
    brand='Topic models comparison',
    brand_href='home',
    color='primary',
    dark=True
    )

app.layout = dbc.Container([
    navbar,
    dash.page_container
],
    fluid=True
)

if __name__ == '__main__':
    app.run_server(debug=True)

