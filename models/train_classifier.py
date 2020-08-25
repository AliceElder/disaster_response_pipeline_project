import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.neighbors import KNeighborsClassifier


def load_data(database_filepath):
    '''
    This function loads data from a database into a dataframe

    Parameters:
    database_filepath: the database containing the data

    Returns:
    X: the values in the 'Messages' column (explanatory variable)
    Y: the values in the target columns (predictive variable)
    category_names: the names of the categorical columns
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages',con=engine)
    engine.dispose()

    X = df['message']
    y = df[df.columns[4:]]
    categories = y.columns.tolist()
    print(categories)
    print(y.head())

    return X, y, categories


def tokenize(text):
    '''
    This function cleans text and splits it into a set of tokens

    Parameters:
    text: the text to be cleaned

    Returns:
    clean_tokens: the cleaned text
    '''

    #Strip punctuation and convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function creates a machine learning pipeline

    Parameters:
    None

    Returns:
    pipeline: model object
    '''
    #clf = AdaBoostClassifier()

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the model and prints classification results

    Parameters:
    model: the model to be evaluated
    X_test: the test set df
    Y_test: the test classifications df
    category_names: the list of categoy names

    Returns:
    None
    '''

    y_pred = model.predict(X_test)
    #print(y_pred)
    #print(Y_test.head())
    #labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(Y_test.values.argmax(axis=1), y_pred.argmax(axis=1), labels=[0,1])
    accuracy = (y_pred == Y_test).mean()
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(Y_test,y_pred,target_names=category_names))


def save_model(model, model_filepath):
    '''
    This function creates a machine learning pipeline

    Parameters:
    model: the model to be evaluated
    model_filepath:

    Returns:
    None
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


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
