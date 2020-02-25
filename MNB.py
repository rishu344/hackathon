import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import re

df = pd.read_excel('new.xlsx')

corpus = []

for i in range(0, 75000):
    review = re.sub('[^a-zA-Z]', ' ', df['Feedback'][i])
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review
              if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(corpus).toarray()

#y = df.iloc[:, 1].values
y = df.iloc[:75000, 2].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Multinomial NB

# Fitting Naive Bayes to the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=510)
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, pred)
print(score)

import pickle

with open('model.pickle', 'wb') as f:
    pickle.dump(classifier, f)

with open('tfidfCV.pickle', 'wb') as f:
    pickle.dump(tfidf, f)

