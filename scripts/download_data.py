from utils.downloading import DataDownloader

downloader = DataDownloader(verbose=True)

subreddits = ['AskWomen', 'AmITheAsshole']

for subreddit in subreddits:
    downloader.download_data(subreddit, saveas=True, return_df=False, start_date="2020-10-01")