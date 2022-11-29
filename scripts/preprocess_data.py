from utils.preprocessing import DataPreprocessor


def preprocess_data(subreddits, columns_df):
    preprocessor_lem = DataPreprocessor(lematize=True, stem=False)
    preprocessor_stem = DataPreprocessor()

    for subreddit in zip(subreddits, columns_df):
        df = preprocessor_lem.read_data(subreddit[0])
        df_lem = preprocessor_lem.preprocess_dataframe(df, text_column=subreddit[1],
                                                       dest_column='lematized', remove_empty_rows=True)
        df_stem = preprocessor_stem.preprocess_dataframe(df_lem, text_column=subreddit[1],
                                                         dest_column='stemmed', remove_empty_rows=True)
        processed_df = df_stem.loc[:, ['lematized', 'stemmed', 'date']]
        preprocessor_lem.save(processed_df, subreddit[0])

subreddits = ['Movies']
columns_df = [['title', 'text']]

preprocess_data(subreddits, columns_df)