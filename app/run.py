import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

#from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib

from plotly.graph_objs import Bar
from plotly.graph_objs import Pie


app = Flask(__name__)

def tokenize(text):
    '''
    INPUT 
        text - a text message entered by the user on the app webpage
        
    OUTPUT
        clean_tokens - a list of tokens extracted from the text input

    Takes a text message, makes text lowercase, removes leading and trailing blank spaces,
    and uses a tokenizer and a lemmatizer to produce a list of tokens
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data from the database to a dataframe
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Data', engine)

# load the classifier model
model = joblib.load("../models/classifier.pkl")


# index webpage displays three visuals and receives user input text for the classification model
@app.route('/')
@app.route('/index')
def index():

    # first chart plots the distribution of message genre
    
    # extract data needed for graph-1
    # get the counts of messages per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create a list to keep the visuals
    graphs = []

    # create visuals for graph-1
    graph_one = []
    # create a Bar chart
    graph_one.append(Bar(
        x=genre_names,
        y=genre_counts))
    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre'),
                yaxis = dict(title = 'Count'))


    # second chart plots the count of messages for each category
    
    # extract data needed for graph-2
    # calculate the counts of labels 0 and 1 for every message category 
    df_counts = df[list(df.columns[4:])].apply(pd.Series.value_counts)
    # select the counts for label equal to 1 for each category
    s = df_counts.loc[1].sort_values(ascending=False)

    categories_values1 = list(s.values)
    categories_names = list(s.keys())

    # create visuals for graph-2
    graph_two = []
    # create a Bar chart
    graph_two.append(Bar(
            x = categories_values1,
            y = categories_names,
            orientation='h',
            name='1',
            marker=dict(
                color='rgba(255, 191, 0, 0.6)', 
                line=dict(color='rgba(255, 191, 0, 1.0)', width=3)
            )
            ))

    
    layout_two = dict(title = 'Count of Message Categories',
                height = 1000,
                xaxis = dict(title = 'Count'),
                yaxis = dict(title = 'Category', automargin = True))

    # third chart plots the percentage of messages for each category for messages
    # that are labeled 'related'

    # extract data needed for graph-3
    # select only the messages that are labeled 'related'
    df_related = df[df['related'] == 1]
    # calculate the counts of labels 0 and 1 for every message category other than related
    df_related_counts = df_related[list(df.columns[5:])].apply(pd.Series.value_counts)
    # select the counts for label equal to 1 for each category
    s2 = df_related_counts.loc[1].sort_values(ascending=False)
    
    related_categories_values1 = list(s2.values)
    related_categories_names = list(s2.keys())

    # create visuals for graph-3
    graph_three = []
    # create a Pie chart
    graph_three.append(Pie(
            values = related_categories_values1,
            labels = related_categories_names,
            ))

    
    layout_three = dict(title = 'Percentage of Message Categories for Messages labeled \'Related\'',
                        height = 800)


    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))

    # plot ids for the html id tag
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    
    # Convert the plotly figures to JSON for javascript in html template
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
