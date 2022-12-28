import pandas as pd
import praw
import os
from psaw import PushshiftAPI
import datetime as dt
import warnings
import definitions as d
import threading

warnings.filterwarnings('ignore')


class DataDownloader:
    """
    Download posts from chosen subreddits and time range via Pushshift API
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def api_error(self):
        raise Exception("Pushshift API connection timeout")

    def download_data(self, subreddit, start_date=d.START_DATE, end_date=d.END_DATE,
                      columns=None, saveas=False, return_df=True):
        """
        Download and save the posts from subreddits

        Parameters
        ----------
        subreddit : str
            subreddit to download
        start_date: DateTime
            first date from which the post should be downloaded
        end_date: DateTime
            last date to which the post should be downloaded
        columns: list of str
            columns to be included in the downloaded dataframe, be default 'title', 'selftext', 'created'
        saveas: bool
            choose whether to save the result dataframe in the raw directory
        return_df: bool
            choose whether to return the DataFrame object as a result of a function invoke
        """

        if columns is None:
            columns = ['title', 'selftext', 'created']

        reddit = praw.Reddit(client_id='ha7wPvCUY_DurA', #  H_e4xc9p7tEXUoYj7BjLrw kZUFktG61cWUzjP6CbX1dg
                             client_secret='B98E34rnXS-Qw6Fg4eOwHUpIupQ', #  -f0UwwoDqiODOZhhx48WuJG_tCKOCg FsBhq-JaYn79pFhMOiOtW8FGwoQN7A
                             user_agent='aita_scrapper'# aita_scrapper news_scrapper_app
                             )

        wait = 30
        alarm = threading.Timer(wait, self.api_error)
        alarm.start()
        api = PushshiftAPI(reddit)
        alarm.cancel()

        rem_or_del = ['[removed]', '[deleted]']
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        end_date = int(end_date.timestamp())
        start_date = int(start_date.timestamp())
        next_epoch = start_date + 86400
        counter = 0
        df = pd.DataFrame(columns=columns)
        for i in range((end_date - start_date) // 86400):
            for post in list(
                    api.search_submissions(before=next_epoch, after=start_date, subreddit=subreddit, limit=200,
                                           filter=columns)):
                if post.selftext in rem_or_del:
                    continue
                post = [[getattr(post, att) for att in columns]]
                post_df = pd.DataFrame(post, columns=columns)
                df = pd.concat([df, post_df])
                counter += 1

            if self.verbose:
                print('day ' + str(i) + ':\n')
                print('processed ' + str(counter) + ' entries')

            start_date += 86400
            end_date += 86400
            next_epoch += 86400

        if 'created' in df.columns:
            df.created = df.created.apply(lambda x: str(dt.date.fromtimestamp(x)))
        # if 'title' in columns:
        #     df.title = df.title.apply(lambda x: str(x))
        # if 'selftext' in columns:
        #     df.selftext = df.selftext.apply(lambda x: str(x))
        df = df.reset_index(drop=True)
        df = df.rename(columns={"selftext": "text", "created": "date"})
        df = df.sort_values('date')
        path = d.RAW_DIR
        if not os.path.exists(path):
            os.mkdir(path)
        print("test")
        if saveas and type(saveas) == str:
            name = os.path.join(d.RAW_DIR, saveas) + ".csv"
            df.to_csv(name, index=False)
        elif saveas and type(saveas) == bool:
            print('DEBUG')
            name = os.path.join(d.RAW_DIR, subreddit.lower()) + ".csv"
            print(name)
            df.to_csv(name, index=False)
        if return_df:
            return df
        return
