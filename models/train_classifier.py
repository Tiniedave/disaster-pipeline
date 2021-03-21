import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

import pickle



def load_data(database_filepath):
    '''Load dataset from database with [`read_sql_table`]
     - Define feature and target variables X and Y
    
    Args:
    database_filepath
    
    Returns: 
    X, Y, category_names
    '''
    
    engine = create_engine('sqlite:///DisasterResponseProject.db')
    df = pd.read_sql_table('DisasterResponseProjectTable',engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis = 1)
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    '''Tokenizes Text
    - Changes to lowercase
    - Removes puntuation
    - Lemmatize tokens
    Args:
    text
    Returns:
    clean_tokens
    
    '''
    #changing to lowercase and removal of puntuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    #tokenize text
    tokens = word_tokenize(text)
    
    # initalize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())
        
    return clean_tokens


def build_model():
    '''Builds a parameter optimized model
    
    Returns:
        cv
    '''
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(MultinomialNB()))])
    
   
    
    parameters = {
        'vect__stop_words':['english'],
        'vect__ngram_range':[(1,2)],
        'vect__max_features':[40000]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test))
    
    for i in range(Y_test.shape[1]):
        print('Column {}, {}'.format(i, category_names[i]))
        print(classification_report(Y_test.iloc[:,i], y_pred.iloc[:,i]))
    return None


def save_model(model, model_filepath):
    '''Saves the model as a pickle file'''
    try:
        f = open(model_filepath, 'wb')
        pickle.dump(model, f)
        return True
    except IOError:
        return False


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()