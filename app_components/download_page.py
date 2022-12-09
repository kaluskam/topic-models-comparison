import dash
from dash import html
import dash_mantine_components as dmc


dash.register_page(__name__, path='/download-subreddits')


layout = html.Div(
    children=[dmc.TextInput(id='text-input-download'),
    html.H1("The data downloading module is not integrated with the app yet. Please use the commandline to download new subreddits.")]
)