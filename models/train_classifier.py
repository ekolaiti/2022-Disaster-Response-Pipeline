# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    INPUT 
        database_filepath - a path to a database file to be loaded
        
    OUTPUT
        X - series with feature variables
        Y - dataframe with target variables
        category_names

    Loads data from a database to a dataframe, and produces a series with the feature variable (message)
    and a dataframe with the target variables (categories).
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # load data from the 'Data' table
    df = pd.read_sql_table('Data', con=engine)

    # define feature variable (message)
    X = df['message']
    # define target variables (categories)
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # get the names of the category columns
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    '''
    INPUT 
        text - a text message that needs to be tokenized
        
    OUTPUT
        tokens - a list of tokens extracted from the text message

    Takes a text message, makes text lowercase, removes leading and trailing blank spaces,
    and uses a tokenizer and a lemmatizer to produce a list of tokens without any English stopwords 
    '''
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stopwords.words("english")]       
    
    return tokens


def build_model():
    '''
    INPUT 
        None
        
    OUTPUT
        cv - a classification model

    Builds a classification model using a pipeline.
    The pipeline uses CountVectorizer and TfidfTransormer to generate features.
    Also it uses MultiOutputClassifier (along with Adaboost as a predictor ML algorithm) to predict multiple target variables.
    Lastly it uses Grirsearch to test different hyper parameters to optimize the model.
    '''    
  
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(AdaBoostClassifier(random_state = 42))),
    ])


    # define parameters for GridSearchCV
    parameters = {
        'moc__estimator__n_estimators': [50, 100],
        #'moc__estimator__random_state': [0, 1],
        'moc__estimator__learning_rate': [0.001, 0.01, 0.1, 1]
    }


    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT 
        model - a classification model
        X_test - a series with the feature testing data (message)
        Y_test - a dataset with the target variables testing data (categories)
        category_names - the names of the columns for the target variables (categories)
        
    OUTPUT
        None

    Prints out the best parameters, f1 score, precision and recall for the test set for each category.
    '''
    Y_pred = model.predict(X_test)

    print("Best Parameters:", model.best_params_)
    print("Best score:", model.best_score_)
    print(classification_report(Y_test, pd.DataFrame(Y_pred, columns=category_names), target_names=category_names))


def save_model(model, model_filepath):
    '''
    INPUT 
        model - a classification model
        model_filepath - file path where the model is to be saved
        
    OUTPUT
        None

    Exports the classification model as a pickle file in the given filepath
    '''
    
    # Export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print('X\n', X)
        print('Y\n', Y)
        print('category_names\n', category_names)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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
