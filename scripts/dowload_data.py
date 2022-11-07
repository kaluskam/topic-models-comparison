from utils.downloading import DataDownloader

downloader = DataDownloader()

subreddits = ['AskMen', 'AskWomen', 'AmITheAsshole']

for subreddit in subreddits:
    downloader.download_data(subreddit, saveas=True, return_df=False)