import dash
from dash import html

dash.register_page(__name__, path='/download-subreddits')

layout = html.Div(
    html.H1("The data downloading module is not integrated with the app yet. Please use the commandline to download new subreddits.")
)