import pandas as pd
import praw
from psaw import PushshiftAPI
import datetime as dt
import warnings
import copy
warnings.filterwarnings('ignore')


class DataDownloader:
    def __init__(self, subreddit, start_date="2020-01-01", end_date="2022-01-01",
                 columns=None, verbose=False):

        if columns is None:
            columns = ['id', 'title', 'score', 'num_comments', 'selftext', 'created', 'link_flair_text']
        self.subreddit = subreddit
        self.start_date = start_date
        self.end_date = end_date
        self.columns = columns
        self.verbose = verbose

    def download_data(self, saveas=False):

        reddit = praw.Reddit(client_id='ha7wPvCUY_DurA',
                             client_secret='B98E34rnXS-Qw6Fg4eOwHUpIupQ',
                             user_agent='aita_scrapper'
                             )
        api = PushshiftAPI(reddit)
        rem_or_del = ['[removed]', '[deleted]']
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        end_date = int(end_date.timestamp())
        start_date = int(start_date.timestamp())
        next_epoch = start_date + 86400
        counter = 0
        aita_df = pd.DataFrame(columns=self.columns)
        for i in range((end_date - start_date) // 86400 + 1):
            for post in list(
                    api.search_submissions(before=next_epoch, after=start_date, subreddit=self.subreddit, limit=200,
                                           filter=self.columns)):
                if post.selftext in rem_or_del:
                    continue
                post = [[getattr(post, att) for att in self.columns]]
                post_df = pd.DataFrame(post, columns=self.columns)
                aita_df = pd.concat([aita_df, post_df])
                counter += 1

            if self.verbose:
                print('day ' + str(i) + ':\n')
                print('processed ' + str(counter) + ' entries')

            start_date += 86400
            end_date += 86400
            next_epoch += 86400

        if 'created' in aita_df.columns:
            aita_df.created = aita_df.created.apply(lambda x: str(dt.date.fromtimestamp(x)))
        aita_df = aita_df.reset_index(drop=True)

        if saveas:
            name = "../data/" + saveas + ".csv"
            aita_df.to_csv(name, index=False)

        return aita_df
