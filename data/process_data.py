import sys
import pandas as pd
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

    # Load data from CSVs
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge dataframes
    df = messages.merge(categories, on = 'id')

    return df


def clean_data(df):
    '''
    This function cleans the data for processing

    Parameters:
    df: a dataframe containing the data to be cleaned

    Returns:
    df: the dataframe containing the cleaned data
    '''

    categories = df['categories'].str.split(';',expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[[0]]

    # Get the column names from the data
    colnames = []
    for col in row.iteritems():
        colname = col[1].str.split('-')[0][0]
        colnames.append(colname)

    # Rename the columns of `categories`
    categories.set_axis(colnames, axis='columns', inplace=True)

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str.replace(r'[^\d.]+','')

        # Replace any 2s in the dataset with 1s
        categories[column] = categories[column].astype('str').str.replace('2', '1')

        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop related column as it does not contain any useful data
    categories = categories.drop(columns=['related'])

    # Drop rows where all the values are 0 or nan
    categories = categories[(categories.T != 0).any()]
    categories = categories.dropna()

    # Drop columns where all the values are now the same (i.e. the col has a std of 0)
    categories = categories.drop(categories.std()[(categories.std() == 0)].index, axis=1)

    # Drop the original categories column from df
    df = df.drop(columns=['categories'])

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1,join='inner')

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    This function saves the data to a SQL db

    Parameters:
    df: a dataframe containing the data to be saved
    database_filename: the location of the SQL db

    Returns:
    None
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    engine.dispose()


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
