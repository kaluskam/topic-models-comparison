# Script downloading data from given subreddits
from utils.downloading import DataDownloader

downloader = DataDownloader()
subreddits = ['AmITheAsshole', 'AskMen', 'AskWomen']


for subreddit in subreddits:
    downloader.download_data(subreddit, start_date="2020-10-01", saveas=True, return_df=False)