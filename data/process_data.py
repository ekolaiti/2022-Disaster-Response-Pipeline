import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT 
        messages_filepath - the filepath to the messages csv
        categories_filepath - the filepath to the categories csv
        
    OUTPUT
        df - a dataframe with the data from messages.csv and categories.csv

    Takes the filepaths to the csv files and returns a dataframe with the data merged together
    using the common id
    '''
        
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    # note that with an outer merge you keep the data that are not common in the two datasets
    df = messages.merge(categories, how='outer', on=['id'])
    df.head()
   
    return df

    
def clean_data(df):
    '''
    INPUT 
        df - a dataframe holding a column that needs cleaning
        
    OUTPUT
        df - a clean dataframe that can be used in a classification model

    Takes a dataframe, separates data in columns, ensures every target column has
    binary labels (categories) and removes duplicates. 
    '''

    # get a dataframe with the category data
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    # note that the first row need to be a Series not a dataframe
    row = categories.iloc[0]
    
    # use the first row to extract a list of new column names for categories
    # note that apply works with a Series not a dataframe
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast = 'integer')

     # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)   

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    
    # drop the column that has only one label 'child_alone': [0]
    #df = df.drop(['child_alone'], axis=1)

    # update the rows of the 'related' column so that 2 becomes 1 
    df.loc[df['related'] == 2, 'related'] = 1

    # drop any duplicate messages
    df.drop_duplicates(subset=['message'], inplace=True)

    return df

def save_data(df, database_filename):
    '''
    INPUT 
        df - a dataframe
        database_filename - a name for the database file to be saved
        
    OUTPUT
        None

    Saves a dataframe to a database with the given name 
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save data to the 'Data' table, drop the table before inserting new values.
    df.to_sql(name='Data', con=engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # read in csv files
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df)

        # clean data
        print('Cleaning data...')
        df = clean_data(df)
        print(df)

        # load to database
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
