from utils.downloading import DataDownloader

downloader = DataDownloader(verbose=True)

subreddits = ['Movies']

for subreddit in subreddits:
    downloader.download_data(subreddit, saveas=True, return_df=False, start_date="2019-10-01", end_date = "2022-09-30")