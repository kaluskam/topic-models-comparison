import dash
from dash import html, dash_table
from dash.dash_table.Format import Format, Scheme, Trim
import dash_mantine_components as dmc

from metrics.coherence_metric import *
from metrics.diversity_metric import *
from metrics.significance_metric import *
from metrics.similarity_metric import *

dash.register_page(__name__, path='/metrics')
model_names = ["NMF", "LDA"]
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
    #WECoherenceCentroidMetric(),
    TopicDiversityMetric(),
    InvertedRBOMetric(),
    LogOddsRatioMetric(),
    WordEmbeddingsInvertedRBOMetric(),
    WordEmbeddingsInvertedRBOCentroidMetric()
]

tooltip_text = [{"index":
                {
                    'value': '**{}** \n\n {} \n\n Takes values from {} to {} \n\n {} is better.'.format(metric.__class__.__name__,
                                       metric.description,
                                       str(metric.range[0]),
                                       str(metric.range[1]),
                                       "**Higher**" if metric.flag else "**Lower**"
                                       ),
                    'type' : 'markdown'
                }} for metric in metrics]

metrics_df = pd.read_csv("scores_df.csv", index_col = 0).transpose()
metrics_df = metrics_df.reset_index()

layout = html.Div([
    dmc.Space(h=10),
    dash_table.DataTable(metrics_df.to_dict('records'),
 [{"name": i, "id": i, "type": 'numeric', "format": Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes)} for i in metrics_df.columns],
 tooltip_data = tooltip_text,
 tooltip_delay=0,
 tooltip_duration=None)
], style = {'width': '50%',
            'margin-left': 'auto',
            'margin-right': 'auto'}
)