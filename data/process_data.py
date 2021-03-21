import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Loads ands merge message and categories csv files
     
    Args:
        message_filepath (str)
        categories_filepath (str)
    
    Returns:
        df (pandas.DataFrame)
    '''
    
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on = 'id')
    return df




def clean_data(df):
    '''Cleans the Disaster Reponse Pipeline dataframe
        - Splits the values in the categories column on the ; character so that each value becomes a separate column.
        - Uses the first row of categories dataframe to create column names for the categories data
        - Converts category values to just numbers 0 or 1
        - Drops duplicates
    Args:
        df (pandas.DataFrame)
    Returns:
        df (pandas.DataFrame): cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns

    categories = df['categories'].str.split(pat = ";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[[1]]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [i.split('-')[0] for i in row.values[0]]
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # drop the original categories column from `df`
    df.drop(['categories'],axis = 1,inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe

    df = pd.concat([df,categories],axis = 1)

    df.drop( df[ df['related'] == 2 ].index , inplace=True)

    # drop duplicates

    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''Save the clean dataset into an sqlite database
    
    Args:
    df (pandas.DataFrame)
    database_filename (str)
    
    Returns: 
    None
    '''
    engine = create_engine('sqlite:///DisasterResponseProject.db')
    df.to_sql('DisasterResponseProjectTable', engine,if_exists='replace', index=False) 


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