# Disaster Response Pipeline Project

![Frontend](/app/images/indexPage1.png)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Training](#training)
	4. [Usage](#usage)
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

1. A distribution of message types

![Sample Input](/app/images/Plot1.png)

2. A distribution of message categories

![Sample Input](/app/images/Plot2.png)

