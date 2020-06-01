# Disaster Response Pipeline Project

![Frontend](screenshots/intro.png)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Training](#training)
    4. [Usage](#usage)
	5. [Extras](#extras)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#screenshots)

<a name="descripton"></a>
## Description

This project is a NLP message classification project that seggregates messages into seperate disaster based categories. The data has numerous tweets/messages that were sent at the time of disaster. The trained classifier will help emergency workers to classify disaster events and send out aid accordingly. 

This project is part of Data Science Nano Degree from Udacity in colaboration with <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a>

The projects has 3 important parts:
1. ETL Pipeline - Extracts data from source, clean data and strore it in a usable sqlite format. 
2. ML Pipeline - Creates a model trained on the cleaned data to predict the category of disaster response message.
3. Frontend - A flask based web frontend for emergency workers to use the model.

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python version used to make project: 3.6.9
* ML Libraries: SciPy, Pandas, NumPy, Sciki-Learn
* NLP Library: NLTK
* Frontend: Flask, Plotly
* Database: SQLalchemy

<a name="installing"></a>
### Installing
Get a local copy - 
```
git clone https://github.com/PranavPuranik/Disaster-Response-ML-Pipeline.git
```
<a name="Training"></a>
### Training:
Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

<a name="Usage"></a>
### Usage:
Run the following command in the app's directory to run your web app.
    `python run.py`

And the open http://0.0.0.0:3001/

<a name="extras"></a>
### Extras

In the **data** and **models** folder you can find two jupyter notebook that will help you understand how the model works step by step:
1. **ETL Preparation Notebook**: learn everything about the implemented ETL pipeline
2. **ML Pipeline Preparation Notebook**: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn

You can use **ML Pipeline Preparation Notebook** to re-train the model or tune it through a dedicated Grid Search section.
In this case, it is warmly recommended to use a Linux machine to run Grid Search, especially if you are going to try a large combination of parameters.
Using a standard desktop/laptop (4 CPUs, RAM 8Gb or above) it may take several hours to complete. 

<a name="authors"></a>
## Authors

* [Pranav Puranik](https://github.com/PranavPuranik)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/)
* [Figure Eight](https://www.figure-eight.com/)

<a name="screenshots"></a>
## Screenshots

1. This is an example of a message you can type to test Machine Learning model performance

![Sample Input](screenshots/sample_input.png)

2. After clicking **Classify Message**, you can see the categories which the message belongs to highlighted in green

![Sample Output](screenshots/sample_output.png)

3. The main page shows some graphs about training dataset, provided by Figure Eight

![Main Page](screenshots/main_page.png)
