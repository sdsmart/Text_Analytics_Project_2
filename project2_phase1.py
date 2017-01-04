#!/usr/bin/env python3

###########################
##   PROJECT 2 PHASE 1   ##
##  NAME: STEPHEN SMART  ##
###########################

# ---------------
# --- Imports ---
# ---------------
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA
import pandas
import matplotlib.pyplot as plt

# -----------------
# --- Constants ---
# -----------------
data_file = 'srep00196-s2.csv'
num_clusters = 7
cluster_colors = ['#BA0000',
				  '#000399',
				  '#F2E85A',
				  '#00BF39',
				  '#DE37CA',
				  '#60DBDB',
				  '#FFB300']

# --------------
# --- Script ---
# --------------

# Reading the data_file defined above, tokenizing the file into a 'document' of 'terms'
# or in this case a meal of ingredients.
# 'meals' is simply a list of strings where each string represents a two-item meal from
# the file.
meals = []
with codecs.open(data_file, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        if line[0] == '#':
    	    continue
        split_line = line.split(',')
        meals.append(split_line[0] + ' ' + split_line[1])

# Initializing the term frequency inverse document frequency vectorizer and the
# KMeans model to process and cluster our data
# NOTE: Paremeter choices are discussed in detail in the README document
vectorizer = TfidfVectorizer(min_df=75)
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100)

# Vectorizing the meal data into a tfidf matrix
# then applying the kmeans algorithm to cluster the data
tfidf_matrix = vectorizer.fit_transform(meals)
kmeans.fit(tfidf_matrix)

# Getting the clusters and organizing the data using pandas.DataFrame for convenience
# NOTE: The use of pandas here was based on the insights of the following tutorial that
# was posted in the discussion form for project 2: http://brandonrose.org/clustering
clusters = kmeans.labels_.tolist()
data = { 'meals': meals, 'clusters': clusters}
frame = pandas.DataFrame(data, index = [clusters] , columns = ['meals', 'clusters'])

# Performing dimensionality reduction using Principle Component Analysis
# NOTE: This choice is discussed in detail in the README document
pca = RandomizedPCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Constructing another DataFrame using the result of applying Principle Component Analysis
df = pandas.DataFrame(dict(x=pca_result[:, 0], y=pca_result[:, 1], label=clusters))
groups = df.groupby('label')

# Ordering the ingredients in the clusters based on distance from centroid
# and getting all of the feature names (ingredients)
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
ingredients = vectorizer.get_feature_names()

# Defining the names of each cluster based on the top 3 ingredients closest to the
# centroid of the cluster
cluster_names = []
for i in range(num_clusters):
    name = ''
    for j in order_centroids[i, :3]:
        name += (ingredients[j] + ', ')
    name = name[:-2]
    cluster_names.append(name)

# Plotting the resulting data
# NOTE: This plotting code was written based on the plot demonstrated in the following
# link: http://brandonrose.org/clustering. I carefully reviewed the code and am using it
# as a guideline, I did not simply copy and paste the code.
figure, axis = plt.subplots(figsize=(16, 9))
axis.margins(0.10)

for name, group in groups:
    axis.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=cluster_names[name],
    	    color=cluster_colors[name], mec='none')
    axis.set_aspect('auto')
    axis.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
    axis.tick_params(axis= 'y', which='both', left='off', top='off', labelleft='off')
    
axis.legend(numpoints=1)
    
plt.show()

# --- END SCRIPT ---