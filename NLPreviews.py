# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:48:42 2020

@author: a
"""

import pandas as pd

df = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t' , quoting = 3)

#cleaning Text 
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#create bag of words model

from sklearn.feature_extraction.text import CountVectorizer
    
cv = CountVectorizer(max_features = 600)
X = cv.fit_transform(corpus).toarray()

y = df.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



 #Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)













