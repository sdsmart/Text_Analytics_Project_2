#!/usr/bin/env python3

###########################
##   PROJECT 2 PHASE 1   ##
##  NAME: STEPHEN SMART  ##
###########################

# ---------------
# --- Imports ---
# ---------------
import codecs
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
import numpy

# -----------------
# --- Constants ---
# -----------------
data_file = 'yummly.json'

# --------------
# --- Script ---
# --------------
print('\nloading data...\n')

# Reading the data from the data file speciied above into a list of meals
# where each element in the list is a dictionary containing the data
# for a meal.
meals = []
with codecs.open(data_file, encoding='utf-8', errors='replace') as f:
    json_objects = f.read()[2:-2].split('},')

    for i, jo in enumerate(json_objects):
        if i < (len(json_objects) - 1):
    	    jo += '}'

        meal = json.loads(jo)
        meals.append(meal)

# Separating the data into two distinct lists used for classification below
# where data holds strings containing the ingredients for a meal and classes
# hold the cuisine types associated with each meal in data.
data = []
classes = []
for m in meals:
    meal = ''
    for i, ing in enumerate(m['ingredients']):
    	meal += (ing.replace(' ', '_') + ' ')
    meal = meal[:-1]
    data.append(meal)
    classes.append(m['cuisine'])

# Performing feature tranformation using sklearn's TfidfVectorizer
# parameter choices are discussed in detail in the README.txt file.
vectorizer = TfidfVectorizer(min_df=25, max_df=0.5)
data = vectorizer.fit_transform(data)

# Creating a classification model using sklearn's LogisticRegression algorithm
# algorithm choice and the training / testing process is discussed in detail
# in the README.txt file.
classifier = LogisticRegression()
classifier.fit(data, classes)

# Prompting the user to enter the list of ingredients that they are interest in.
print('Please enter all of the ingredients that you would like to use. (separated by spaces)')
print('For ingredients that are more than one word, such as \'ground beef\', please use underscores.\n')
print('Example: salt pepper ground_beef bread oil\n')
ingredients = input('> ')

# Setting up some inital variables that will be used to vectorize the input.
possible_ingredients = vectorizer.get_feature_names()
vectorized_input = [0] * len(possible_ingredients)

# Vectorizing the input so that predictions using the classification model
# can be made.
ingredients = ingredients.split(' ')
for ing in ingredients:
	if ing in possible_ingredients:
		vectorized_input[possible_ingredients.index(ing)] = 1
vectorized_input = numpy.array(vectorized_input).reshape(1, -1)

# Predicting the cuisine type using the LogisticRegression classifier created above.
predicted_cuisine = classifier.predict(vectorized_input)[0]

# Calculating the distances between the ingredients that were entered by the user
# and all of the meals in the dataset.
distances = pairwise_distances(vectorized_input, data).tolist()[0]

# Selecting the closest meals according to the distances calculated above.
closest_meal_indexes = sorted(range(len(distances)), key=lambda k: distances[k])[:5]

# Printing the results (nicely formatted) to the user.
print('\nA meal consisting of the ingredients that you have entered is most likely ' +
	  'of the following cuisine:\n\n----------\n{}\n----------\n'.format(predicted_cuisine))

print('Here are five meals that were found to be similar to a meal consisting ' +
	  'of the ingredients that you entered:\n')

for i in range(len(closest_meal_indexes)):
	meal = meals[closest_meal_indexes[i]]
	print('ID: {}'.format(meal['id']))
	print('Cuisine: {}'.format(meal['cuisine']))
	ingredients = ''
	for ing in meal['ingredients']:
		ingredients += (ing.replace(' ', '_') + '\n')
	print('Ingredients:\n{}'.format(ingredients))

# --- END SCRIPT ---