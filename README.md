# 2022-Disaster-Response-Pipeline
### Description:
In this project, I analyse disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.
I create a machine learning pipeline that categorises the messages so that these can be sent to appropriate disaster relief agencies.
The project includes a web app where a user can input a new message and get classification results in several categories. The web app also displays three visualisations of the training data: 

graph one is a bar chart that shows the distribution of the messages per genre (direct, news, social)

![graph one](https://user-images.githubusercontent.com/50168917/198672567-768c1096-1ade-4a46-924c-7dde9738857f.png)

graph two is a horizontal bar chart that displays a count of the messages per category

![graph two](https://user-images.githubusercontent.com/50168917/198672625-e21a4735-c900-4846-b2f1-d4311c632fc4.png)

graph three is a pie chart that shows the percentage of message categories for messages that are labeled related

![graph three](https://user-images.githubusercontent.com/50168917/198672644-0286bb1d-43e5-476d-bf4f-47a190a70369.png)


### Files:
* This repository includes besides this README.md the following files:
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 


### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run the ETL pipeline that cleans the data and stores in the DisasterResponse database:
       python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run the ML pipeline that trains the classifier and saves the classifier in a pickle file:
       python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app: python3 run.py

3. Go to http://0.0.0.0:3001/ or http://127.0.0.1:3001/ 

### Considerations:
This dataset is imbalanced, some labels have too few examples of the minority class (label = 0) for the model to effectively learn the decision boundary. One example is the ‘child alone’ category which hasn’t got any examples (with label = 1). As a result using an SVC classifier throws the following error: “ValueError: The number of classes has to be greater than one; got 1 class”. 
This is why I use a tree based model instead. I use Adaboost for the predictor and it worked returning the following best parameters: {'moc__estimator__learning_rate': 1, 'moc__estimator__n_estimators': 100}


### Acknowledgments:
* A massive thank you to Appen for providing access to the data and to Udacity for designing this project.
