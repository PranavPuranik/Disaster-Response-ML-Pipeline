import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
	"""
	Loads data from sqlite database and divides it into labels and target.

	Input: sqlite database filepath

	Output: X - Feature Array
			Y - Target Array
			category_names - names of categories in Y
	"""
	engine = create_engine('sqlite:///' + database_filepath)
	df = pd.read_sql_query('SELECT * FROM DisasterResponseTable', engine)

	X = df['message']
	Y = df.iloc[:, 4:]
	category_names = Y.columns.values

	return X.values, Y.values, category_names


def tokenize(text):
	"""
	Returns tokenized->lemmatized->case_lowered text

	Input: text in string format

	Output: list of normalized tokens
	"""
	
	tokens = nltk.word_tokenize(text)
	lemmatizer = nltk.WordNetLemmatizer()

	clean_tokens = []
	for tok in tokens:
		clean_tok = lemmatizer.lemmatize(tok).lower().strip()
		clean_tokens.append(clean_tok)

	return clean_tokens


def build_model():
	"""
	Making a pipeline and using grid-search to optimize it.

	Input: None

	Output: Returns a model
	"""
	
	pipeline = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(RandomForestClassifier()))
	])

	parameters = {
		'vect__max_df': [0.5, 1.2],
		'clf__estimator__n_estimators': [10, 50],
		'clf__estimator__min_samples_split': [2, 3, 4],
		'clf__estimator__criterion': ['entropy', 'gini']
	}

	return GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs = -1)

def evaluate_model(model, X_test, Y_test, category_names):
	"""
	Displays precision, recall, f1-score, and support.

	Input: 	model
			X_test - Features from test set
			Y_test - Target values of X_test
			category_names - list of target values

	Output: None, prints the scores.
	"""
	
	Y_predict = model.predict(X_test)
	Y_predict_t = Y_predict.T

	Y_actual = Y_test.T

	for i, pred in enumerate(Y_predict_t):
		print(category_names[i])
		print(classification_report(Y_actual[i], pred))

def save_model(model, model_filepath):
	"""
	Saves model as pickle file.

	Input: model, location
	Output: None, saves model to the location.
	"""
	
	with open(model_filepath, 'wb') as file:
		pickle.dump(model, file)


def main():
	'''
	Main function that loads data, trains model, and saves it.
	'''

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