from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from app_components.modelling_page import modelling_page

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink('Modelling', href='modelling')),
        dbc.NavItem(
            dbc.NavLink('Model comparison', href='comparison')),
        dbc.NavItem(
            dbc.NavLink('Data Exploration', href='data-exploration'))
        ],
    brand='Topic models comparison',
    brand_href='home',
    color='primary',
    dark=True

    )

app.layout = dbc.Container([
    navbar,
    modelling_page
],
    fluid=True
)


if __name__ == '__main__':
    app.run_server(debug=True)

