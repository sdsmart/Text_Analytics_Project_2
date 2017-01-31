Meal Clustering & Prediction
----------------------------
The purpose of this project is to cluster and visualize various food related data contained 
in some predefined datasets using some different machine learning techniques.

### Run Instructions
Clone the repository and run either the cluster_meals.py or predict_meals.py script.

#### === For predict_meals.py ===
Make sure that the script is in the same folder as the data file: yummly.json

On my machine, this program takes around 10 seconds to load the data, all other
actions are very fast.

This program will ask the user to input a list of ingredients. Directions are
given in the program, but I will re-iterate here:

ingredients entered should be separated by spaces, and multi-word ingredients should
use underscores. Here is an example:

romain_lettuce purple_onion garlic seasoning olive_oil salt mushrooms

#### === For cluster_meals.py ===
Make sure that the script is in the same folder as the data file: srep00196-s2.csv

Note that on my machine, this program takes around one minute or so to complete.

### Design choices
#### === For predict_meals.py ===
##### Clustering (or lack of)
One major design decision was to NOT cluster the yummly data like the srep data.
I felt that the yummly data actually provided perfect clusters for me (the cuisine field).
I decided that there was no need to perform clustering on this data (although I easily
could have), because the data was already perfectly separated into the various cuisine types.
Therefore, all I needed to do, was read in the data and separate the ingredients and the
cuisine types into different structures in order to pass them into the Vectorizer and the
and the classifier.

##### Vectorizing
When vectorizing, I chose a min_df value of 25 because after tuning this value, I felt that 25
eliminated ingredients that I found to carry little meaning. I also chose to set max_df value
of 0.5 (50%) in order to ignore ingredients that occurred too often in the meals. This resulted
in trimming the features by only a few, but I think it was improvement.

##### Classification
I chose to go a little outside the scope of the class and use the LogisticRegression machine
learning algorithm for classification in sklearn. I found it easy to use and effective in
at predictions.

To arrive at this decision, I first split the dataset using sklearn's train_test_split function
by 70% train -- 30% test. I then trained numerous classifiers (such as KNeighbors, SVC, 
and many more) using this data and tested it.

I found that KNeighbors with # neighbors == 20 resulted in roughly 70% accuracy when predicting.
SVC resulted in roughly 75% accuracy when predicting (but ran very slowly)
and LogisticRegression resulted in roughly 75% accuracy when predicting but ran much faster.
The other algorithms that I tested were not over 60%.

I then removed the train_test_split code and trained my classifier over the entire dataset.

##### Calculating Closest Meals
To calculate the closest meals to the ingredients entered by the user, I simply executed
the pairwise_distance function that Dr. Grant linked in the Discussion, sorted the result,
and chose the first 5 meals (the 5 meals with smallest distances).

This seemed to give promising results.

#### === For cluster_meals.py ===
The pipeline for cluster_meals.py is as follows:

data-extraction --> vectorize --> cluster --> dimensionality-reduction --> visualize

Extracting the data was very straight forward, so I will not go into detail about
that process.

I used the sklearn python library to vectorize, cluster, and apply dimensionality
reduction.

##### Vectorizing
When vectorizing, I simply called upong the TfidfVectorizer in sklearn
and passed it in my list of meals. I decided to ignore ingredients that occur in
less than 75 meals in the dataset. Since the dataset is so large (221k meals), I am
still considering most of the ingredients (978 out of 1507). I feel that based on
the visualization, the meals that occur more than about 75 times in the document
are the more important meals. This min_df value also helps reduce the amount of data
so that plotting and doing dimensionaility reduction does not cause memory problems.

There is no need to choose a max_df because there are actually no ingredients that
occur overly frequently in the document, in my opinion. No ingredient occurs in more
than 10 percent of the meals.

I decided to not take use of the max_features paremeter because my min_df is
successfully reducing the data down to what I want to visualize.

##### Clustering
For the clustering, I chose to use the KMeans algorithm because it is the one that
I was most familiar with based on the lectures and reading.

I tweaked the number of clusters to what felt like countless different numbers
and I was never 100% satisfied with the result. I landed on 6 clusters as I feel
it made the most sense for my visualization and for breaking up meals based on
ingredients to learn about the meal.

I will discuss more about my problems in the next section.

##### Dimensionality Reduction
At first I chose to employ Multi-Dimensional Scaling to reduce the complexity of
my data to be able to visualize it. This worked out okay, however I did not like
what the visualization was looking like. Dr. Grant mentioned using a 3D visualization
when using MDS which I tried but got lost and frustrated. After doing some reading
I learned that PCA actually might work better.

I switched over my implementation to employ Principle Component Analysis in order
to reduce complexity and I was more satisified with the result. Also, I no longer
had memory problems when using PCA.

##### Visualization
For cluster names: I chose the 3 most central ingredients in the cluster (3 ingredients
closest to the centroid)

I followed this tutorial: http://brandonrose.org/clustering which helped me learn
a lot about clustering in python and I followed his visualization techniques because
they seemed pretty good! I just want to cite that as a definite source of mine in the
implementation process. (Along with the code that Dr. Grant posted to the discussion)
