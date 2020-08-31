import sys
from sqlalchemy import create_engine
import pandas as pd
import pickle
import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from starting_verb_extractor import StartingVerbExtractor

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedShuffleSplit

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
    # Connect to db
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages',con=engine)
    engine.dispose()

    # Store values
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

    # Strip punctuation and convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Split into tokens and clean
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    This function creates and optimises a machine learning model

    Parameters:
    None

    Returns:
    cv: ML model
    '''

    # Create a pipeline estimator
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Refine model using GridSearch with more optimal params
    parameters = [
        {'features__text_pipeline__vect__ngram_range': [(1, 2)],
         'clf__estimator__n_estimators': [100]}]

    cv = GridSearchCV(pipeline, param_grid=parameters,cv=3,n_jobs=1,verbose=5)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the model and prints classification results

    Parameters:
    model: the model to be evaluated
    X_test: the test set
    Y_test: the test classifications
    category_names: the list of category names

    Returns:
    None
    '''

    # Perform and output evaluations of the model
    y_pred = model.predict(X_test)
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
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y.iloc[:,1])
        sss = StratifiedShuffleSplit(test_size=0.2)
        sss.get_n_splits(X, Y)
        for train_index, test_index in sss.split(X, Y.iloc[:,1]):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

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
