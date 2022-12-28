import dash
from dash import html, dash_table
from dash.dash_table.Format import Format, Scheme, Trim
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from utils.dashboard_utils import get_data_for_subreddit_select
from utils.data_loading import load_downloaded_data, check_cache_existance, create_output_data_cache_filepath, load_cache_output_data
import definitions as d
from models.berttopic_model import *
from models.lda_model import *
from models.nmf_model import *
import datetime as dt

from metrics.coherence_metric import *
from metrics.diversity_metric import *
from metrics.significance_metric import *
from metrics.similarity_metric import *

dash.register_page(__name__, path='/metrics')

metrics = [
    KLUniformMetric(),
    KLBackgroundMetric(),
    RBOMetric(),
    WordEmbeddingPairwiseSimilarityMetric(),
    WordEmbeddingCentroidSimilarityMetric(),
    PairwiseJacckardSimilarityMetric(),
    UMassCoherenceMetric(),
    CVCoherenceMetric(),
    CUCICoherenceMetric(),
    CNPMICoherenceMetric(),
    WECoherencePairwiseMetric(),
    WECoherenceCentroidMetric(),
    TopicDiversityMetric(),
    InvertedRBOMetric(),
    LogOddsRatioMetric()
]

subreddit_select = html.Div([
    dmc.MultiSelect(
        id='subreddits-multiselect',
        label='Subreddits',
        data=get_data_for_subreddit_select(),
        value=['science']
    )]
)

n_topics = html.Div([
    dmc.TextInput(id='n-topics',
                  label='Number of topics',
                  type='number',
                  value=20)
])


date_range_picker = html.Div([
    dmc.DateRangePicker(id='date-range-picker',
                        label='Date range',
                        minDate=d.START_DATE,
                        maxDate=d.END_DATE,
                        value=[d.END_DATE - dt.timedelta(days=364), d.END_DATE])])

run_analysis_button = html.Div([
    dmc.Button('Run analysis',
               id='run-exploration',
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
        n_topics
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

tooltip_text = [{"Metric Name":
                {
                    'value': '**{}** \n\n {} \n\n Takes values from {} to {} \n\n {} is better.'.format(metric.name,
                                       metric.description,
                                       str(metric.range[0]),
                                       str(metric.range[1]),
                                       "**Higher**" if metric.flag else "**Lower**"
                                       ),
                    'type' : 'markdown'
                }} for metric in metrics]




MODEL_NAMES_DICT = {
            'nmf': NMFModel(),
            'lda': LDAModel(),
            'bertopic': BERTopicModel()
        }

number_topics = 20
subreddits = ["science"]
date_range = [str(d.START_DATE), str(d.END_DATE)]
model_order = []
calculated_metrics = []
for model_name, model in MODEL_NAMES_DICT.items():
    model_order.append(model_name)
    num_topics_for_cache = str(number_topics) if model_name != "bertopic" else "None"
    if len(subreddits) == 1 and check_cache_existance(subreddits[0], date_range, model_name, num_topics_for_cache):
        cache_file = create_output_data_cache_filepath(subreddits[0], date_range, model_name, num_topics_for_cache)
        output = load_cache_output_data(cache_file)
        calculated_metrics.append(output.metrics_dict)
    else:
        print(subreddits[0], date_range, model_name, num_topics_for_cache)
        print(create_output_data_cache_filepath(subreddits[0], date_range, model_name, num_topics_for_cache))

metrics_df = pd.DataFrame(calculated_metrics).transpose()
metrics_df.columns = [name.upper() for name in model_order]
metrics_df = metrics_df.reset_index()
metrics_df.rename(columns={ metrics_df.columns[0]: "Metric Name" }, inplace = True)



tabler = dash_table.DataTable(metrics_df.to_dict('records'),
 [{"name": i, "id": i, "type": 'numeric', "format": Format(precision=3, scheme=Scheme.fixed, trim=Trim.yes)} for i in metrics_df.columns],
 tooltip_data = tooltip_text,
 tooltip_delay=0,
 tooltip_duration=None,
 id = "tabler")



layout = dbc.Container([
    dmc.Space(h=20),
    dmc.Alert(id='alert-metrics',
              title=d.ALERT_TITLE,
              children=d.ALERT_MESSAGE,
              color='red'),
    dmc.Space(h=20),
    dbc.Row([
        dbc.Col(controls_card, md=4),
        dbc.Col(tabler, md = 8)
    ])
],
    fluid=True
)


@callback(
    Output(component_id='run-exploration', component_property='n_clicks'),
    Output(component_id='tabler', component_property='data'),
    Output(component_id='alert-metrics', component_property='hide'),
    Output(component_id="tabler", component_property="tooltip_data"),
    Input(component_id='run-exploration', component_property='n_clicks'),
    Input(component_id='subreddits-multiselect', component_property='value'),
    Input(component_id='n-topics', component_property='value'),
    Input(component_id='date-range-picker', component_property='value'))
def run_analysis(n_clicks, subreddits, n_topics, date_range):
    if n_clicks is not None and n_clicks >= 1:
        
        MODEL_NAMES_DICT = {
            'nmf': NMFModel(),
            'lda': LDAModel(),
            'bertopic': BERTopicModel()
        }

        tooltip_text = [{"Metric Name":
                {
                    'value': '**{}** \n\n {} \n\n Takes values from {} to {} \n\n {} is better.'.format(metric.name,
                                       metric.description,
                                       str(metric.range[0]),
                                       str(metric.range[1]),
                                       "**Higher**" if metric.flag else "**Lower**"
                                       ),
                    'type' : 'markdown'
                }} for metric in metrics]


        model_order = []
        calculated_metrics = []
        for model_name, model in MODEL_NAMES_DICT.items():
            model_order.append(model_name)
            num_topics_for_cache = str(n_topics) if model_name != "bertopic" else "None"
            if len(subreddits) == 1 and check_cache_existance(subreddits[0], date_range, model_name, num_topics_for_cache):
                cache_file = create_output_data_cache_filepath(subreddits[0], date_range, model_name, num_topics_for_cache)
                output = load_cache_output_data(cache_file)
                calculated_metrics.append(output.metrics_dict)
            else:
                print(f"{model_name} this is not cached")
                # input_data = load_downloaded_data(subreddits, date_range)
                # model.fit(input_data, int(num_topics_for_cache))
                # output = model.get_output()
                # output.calculate_metrics(metrics)
                #calculated_metrics.append(output.metrics_dict)
            

            
        metrics_df = pd.DataFrame(calculated_metrics).transpose()
        metrics_df.columns = [name.upper() for name in model_order]
        metrics_df = metrics_df.reset_index()
        metrics_df.rename(columns={ metrics_df.columns[0]: "Metric Name" }, inplace = True)
        
        output = metrics_df.to_dict("records")
        
        return 0, output , True, tooltip_text
    else:
        raise Exception("Doesn't change anything")



