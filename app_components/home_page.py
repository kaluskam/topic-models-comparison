import dash
from dash import html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import dash_core_components as dcc


dash.register_page(__name__, path='')


layout = dbc.Container(
    [
    dbc.Row([
        dmc.Space(h=20),
        dcc.Markdown('''
            ### Topic analysis performed on data from Reddit
            *By: Maria Kędzierska, Marcelina Kurek and Mikołaj Spytek*

            This application is a part of a Bachelor Thesis in the field of Data Science (pol. *Inżynieria i analiza danych*) at the Faculty of Mathematics and Information Science, Warsaw University of Technology.
    ''')],
    ),
    dbc.Row(
        html.Img(src='assets/1-graphical-abstract.png', style = {"width": "60%"}), justify="center"
        ),
    dbc.Row(
        dcc.Markdown('''
        Reddit is a valuable source of texutal data reflecting its users and their priorities. We propose an end-to-end solution to explore topics contained in Reddit posts. This application allows you to extract topics from different subreddits using various modeling techniques, explore the produced topics, as well as compare different methods with each other using different kinds of evaluation metrics.

        The functionalities of this dashboard are contained on different pages available to you in the navigation panel at the top of the page.
        ''')
    )
    ]
    )
    