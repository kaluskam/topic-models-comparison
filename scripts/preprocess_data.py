from utils.preprocessing import DataPreprocessor
import pandas as pd
import pickle

filenames = ['askmen', 'askwomen', 'amitheasshole']


def read_data(files):
    df_dict = {}
    for file in files:
        df_dict[file] = pd.read_csv('../data/' + file + '.csv')
    return df_dict

df_dict = read_data(filenames)
preprocessor = DataPreprocessor(lematize=True, stem=False)
preprocessor_stem = DataPreprocessor()

preprocessed_df = {}
preprocessed_df['askmen'] = preprocessor.preprocess_dataframe(df_dict['askmen'], text_column=['title', 'text'],
                                                   dest_column='processed_text', remove_empty_rows=True)
preprocessed_df['askwomen'] = preprocessor.preprocess_dataframe(df_dict['askwomen'], text_column=['title', 'text'],
                                                     dest_column='processed_text', remove_empty_rows=True)
preprocessed_df['amitheasshole'] = preprocessor.preprocess_dataframe(df_dict['amitheasshole'], text_column='text',
                                                   dest_column='processed_text', remove_empty_rows=True)

with open('../data/preprocessed_text.pkl', 'wb') as f:
    pickle.dump(preprocessed_df, f)

