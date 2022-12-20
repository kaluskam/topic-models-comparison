import os
import definitions as d


def get_available_subreddits():
    """
    Retrieve available (already downloaded) subreddits

    Returns
    -------
    list
        a list of already downloaded subreddits (present in the raw directory data)
    """
    return [subreddit.replace('.csv', '') for subreddit in os.listdir(d.RAW_DIR)]


def get_data_for_subreddit_select():
    """
    Retrieve available subreddits

    Returns
    -------
    dictionary
        a dictionary of downloaded subreddits (present in the raw directory data)
    """
    subreddits = get_available_subreddits()
    result = []
    for subreddit in subreddits:
        result.append({'label': subreddit, 'value': subreddit})
    return result
