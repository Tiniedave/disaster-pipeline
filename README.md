# Disaster Response Pipeline Project

This project uses data provided by Figure Eight to build a model that classifys real messages that were sent during disaster events into various categories. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 
### Install:

This is a list of python libraries needed for this project.
NumPy
Pandas
Nltk
Flask
Sklearn
Sqlalchemy
Sys
Re
Pickle
json
plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
/app/run.py - Python script to run the web app.

/data - Directory for the data files and python script for processing the data.

process_data.py - Python script for loading, cleaning and storing disaster dataset in a SQLite database.

disaster_messages.csv - contains the messages sent during disaster events.

disaster_categories.csv - contains the message categories.

/models/train_classifier.py - Python script for training the classifier and saving the model.

### Acknowledgements:
Dataset used in this project was provided by Figure Eight for Udacity Data Scientist Nanodegree. 
https://www.geeksforgeeks.org/, https://stackoverflow.com/ helped me out when i got stuck.


