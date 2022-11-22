import os
import definitions as d


def get_available_subreddits():
    return [subreddit.replace('.csv', '') for subreddit in os.listdir(d.RAW_DIR)]


def get_data_for_subreddit_select():
    subreddits = get_available_subreddits()
    result = []
    for subreddit in subreddits:
        result.append({'label': subreddit, 'value': subreddit})
    return result