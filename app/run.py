import json
import plotly
import pandas as pd
import joblib
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
#from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_topic = df[['aid_related','infrastructure_related','weather_related']]
    topic_counts = df_topic.sum()
    topic_names = ['Aid','Infrastructure','Weather']
    
    # create counts
    total_messages = df['message'].count()
    total_requests = df['request'].sum()
    percent_requests = "{}% of total messages".format(round(((total_requests/total_messages)*100),1))
    total_offers = df['offer'].sum()
    percent_offers = "{}% of total messages".format(round(((total_offers/total_messages)*100),1))
    
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, 
                    
                )
            ],

            'layout': {
                'title': '<b>Source of Messages</b>'
            }
        },
        {
            'data': [
                Bar(
                    y=topic_names,
                    x=topic_counts,
                    orientation="h",
                    marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, 
                    opacity=0.6
                )
            ],

            'layout': {
                'title': '<b>Topics Mentioned in Messages</b>',
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, total_messages=total_messages, total_requests=total_requests, percent_requests=percent_requests, total_offers=total_offers, percent_offers=percent_offers)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()