import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads data from a two CSVs into a single dataframe

    Parameters:
    messages_filepath: the csv containing the messages data
    categories_filepath: the csv containing the categories data

    Returns:
    df: a dataframe containing the loaded data
    '''

    #Load data from CSVs
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #Merge dataframes
    df = pd.merge(messages, categories, on='id')


def clean_data(df):
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[[0]]

    #Get the column names from the data
    colnames = []
    for col in row.iteritems():
        colname = col[1].str.split('-')[0][0]
        colnames.append(colname)

    # rename the columns of `categories`
    categories.set_axis(colnames, axis='columns', inplace=True)

    for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.replace(r'[^\d.]+', '')

    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)

    # drop duplicates
    df = df.drop_duplicates()


def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
