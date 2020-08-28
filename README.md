# Disaster Response Pipeline Project

### Motivation
This project looks at a set of labeled Disaster Response messages and attempts to categorise any new Messages based on this data.

The project is split into 3 sections:
- Data: the data is combined, cleaned, and exported to a SQLite database.
- Model: a set estimators are applied to the data to produce and output a model.
- App: a web application shows a dashboard of the data, and offers the ability to input and classify new Messages based on the model.

### Results

#### Best Estimators
A number of estimators were tried, and the following were the most successful:
- CountVectorizer:
- TfidfTransformer:
- StartingVerbExtractor: a custom-built class that extracts the first verb from a sentence
- AdaBoostClassifier: a multi-output classifier

#### Best Parameters
A grid search was performed to find the best parameters. The following parameters were the most successful:
| Parameter | Optimal Value |
|:---------:|:-------------:|
|           |               |

#### Accuracy
| Precision | Recall | F1-Score |
|:---------:|:------:|:--------:|
| 0.78      | 0.60   | 0.67     |

### Files
This project is structured as follows:
```bash
├── app
│   ├── template
│   │   ├── master.html                 # main page of web app
│   │   ├── go.html                     # classification result page of web app
│   ├── screenshots
│   │   ├── overview.png                # image of dashboard
│   │   ├── example_classification.png  # image of classification function in app
│   ├── run.py                          # Flask file that runs app
├── data
│   ├── disaster_categories.csv         # data to process 
│   ├── disaster_messages.csv           # data to process
│   ├── process_data.py                 # script to clean and output data to db
│   ├── DisasterResponse.db             # database to save clean data to
├── models
│   ├── train_classifier.py             # script to train model
│   ├── classifier.pkl                  # saved model 
│   ├── starting_verb_extractor         # script to create additional features for classification
└── README.md
```


### Libraries
The following libraries were used:
- pandas
- numpy
- json
- plotly
- joblib
- sys
- nltk
- flask
- sqlalchemy


### Instructions to run the project:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/


### Screenshots
![Overview of dashboard](../master/app/screenshots/overview.PNG)
![Example of classification](../master/app/screenshots/example_classification.PNG)


### Suggestions for Future Improvements
- Several of the columns had very few data points. The accuracy of the model could be improved by collecting more data, in particular for these features:
    - Offers: only 0.3% of fields had a value of 1
- There may be more optimal paramers; GridSearch took an extremely long time to run for more combinations of parameters. 


### Acknowledgements
Thanks to Udacity for providing the files and framework for this project.

#### Data Sources Used
The following data sources are used in this project:
| Data Source Name | Description | Type |
|------------------|-------------|------|
| Messages | A set of Disaster Response messages with English translations | Downloaded csv file |
| Categories | The categorisation of the Disaster Response messages | Downloaded csv file |


### Authors
This project was created by Alice Elder.
