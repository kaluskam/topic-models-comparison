import datetime as dt
import pandas as pd
import os
from utils.data_loading import load_cache_output_data, create_output_data_cache_filepath, load_downloaded_data
from utils.visualizer import plot_topic_overtime_vs_google_trends
import definitions as d


model_aliases = ['nmf', 'lda', 'bertopic']
DATE_RANGE = [dt.date(2019, 10, 1), dt.date(2022, 9, 30)]
filepath = create_output_data_cache_filepath('worldnews',DATE_RANGE,
                                             'nmf', "20")
output = load_cache_output_data(filepath)

for topic in output.get_topics():
    print(topic)
print(output.get_topics()[4])
print(output.get_topics()[9])
print(output.get_topics()[2])
print(output.get_topics()[8])
topics_ids_df = pd.DataFrame(columns=['nmf', 'lda', 'bertopic'])
topics_ids_df.loc['COVID-19 outburst', :] = [4, None, None]
topics_ids_df.loc['COVID-19 vaccines', :] = [9, None, None]
topics_ids_df.loc['Russia invasion of Ukraine', :] = [2, None, None]
topics_ids_df.loc['US 2020 elections', :] = [8, None, None]
print(topics_ids_df)

gt_war_russia_ukraine = pd.read_csv(os.path.join(d.GOOGLE_TRENDS_DIR, 'WarRussiaUkraine.csv'))
plot_topic_overtime_vs_google_trends(output, 2, load_downloaded_data(['worldnews'],
                         DATE_RANGE), gt_war_russia_ukraine, 'War Russia-Ukraine 2022', 'NMF_war_russia_ukrain_gt.jpg')

