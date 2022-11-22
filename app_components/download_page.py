import dash
from dash import html

dash.register_page(__name__, path='/download-subreddits')

layout = html.Div(
    html.H1("Tu będziemy pobierać dane")
)