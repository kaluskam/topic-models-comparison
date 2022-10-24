import pandas as pd
import praw
from psaw import PushshiftAPI
import datetime as dt
import warnings

warnings.filterwarnings('ignore')


class DataDownloader:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def download_data(self, subreddit, start_date="2020-10-01", end_date="2022-10-01",
                      columns=None, saveas=False, return_df=True):

        if columns is None:
            columns = ['title', 'selftext', 'created']

        reddit = praw.Reddit(client_id='ha7wPvCUY_DurA',
                             client_secret='B98E34rnXS-Qw6Fg4eOwHUpIupQ',
                             user_agent='aita_scrapper'
                             )
        api = PushshiftAPI(reddit)
        rem_or_del = ['[removed]', '[deleted]']
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        end_date = int(end_date.timestamp())
        start_date = int(start_date.timestamp())
        next_epoch = start_date + 86400
        counter = 0
        aita_df = pd.DataFrame(columns=columns)
        for i in range((end_date - start_date) // 86400):
            for post in list(
                    api.search_submissions(before=next_epoch, after=start_date, subreddit=subreddit, limit=200,
                                           filter=columns)):
                if post.selftext in rem_or_del:
                    continue
                post = [[getattr(post, att) for att in columns]]
                post_df = pd.DataFrame(post, columns=columns)
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
        aita_df = aita_df.rename(columns={"selftext": "text", "created": "date"})

        if saveas and type(saveas) == str:
            name = "../data/" + saveas + ".csv"
            aita_df.to_csv(name, index=False)
        elif saveas and type(saveas) == bool:
            name = "../data/" + subreddit.lower() + ".csv"
            aita_df.to_csv(name, index=False)
        if return_df:
            return aita_df
        return
