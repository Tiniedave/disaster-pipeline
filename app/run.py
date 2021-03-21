import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
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
engine = create_engine('sqlite:///../data/DisasterResponseProject.db')
df = pd.read_sql_table("DisasterResponseProjectTable", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #count for all messages

    counts = df.iloc[:,4:].apply(lambda x: x.value_counts()).T
    counts.sort_values(by=1, ascending=False, inplace=True)
    social = df.loc[df['genre']=='social']
    
    #count for messages recieved via social
    counts_social = social.iloc[:,4:].apply(lambda x: x.value_counts()).T
    counts_social.sort_values(by=1, ascending=False, inplace=True)
    
    
    categories = pd.Series(counts.index).str.replace('_', ' ').str.title()

    
     #create visuals
     #TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
               Bar(
                    x=genre_names,
                    y=genre_counts
               )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    

    
        {
            'data': [
                Bar(
                    x=categories,
                    y=counts[1],
                    
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -90
                },
                'margin': {
                    'b': 150
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=categories,
                    y=counts_social[1],
                    
                )
            ],

            'layout': {
                'title': 'Distribution of Message on Social Categories',
                'yaxis': {
                    'title': "Social Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -90
                },
                'margin': {
                    'b': 150
                }
            }
        }
    ]
    
        
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
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