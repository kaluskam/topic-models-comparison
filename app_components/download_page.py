import dash
from dash import html

from app_components.drawer import drawer

dash.register_page(__name__, path='/download-subreddits')

layout = html.Div(
    children=[
        html.H1("Tu będziemy pobierać dane")]
)