import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from wordcloud import WordCloud ,STOPWORDS

bbc_text = pd.read_csv(r'\\server\PYTHON\BALARAM PANIGRAHY\BBC - NEWS\bbc-text.csv')

bbc_text.head()
bbc_text.tail()

#Getting all category names
bbc_text['category'].unique()

#Shape of the data set
bbc_text.shape

# checking the data type
bbc_text.dtypes

#Checking of null values
bbc_text.isnull().sum()

#Count plot
sns.countplot(bbc_text.category)

def create_wordcloud(words):
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

#Displaying the word cloud for the business category
subset=bbc_text[bbc_text.category=="business"]
text=subset.text.values
words =" ".join(text)
create_wordcloud(words)

#Displaying the word cloud for the entertainment category
subset=bbc_text[bbc_text.category=="entertainment"]
text=subset.text.values
words =" ".join(text)
create_wordcloud(words)

#Displaying the word cloud for the politics category
subset=bbc_text[bbc_text.category=="politics"]
text=subset.text.values
words =" ".join(text)
create_wordcloud(words)

#Displaying the word cloud for the sport category
subset=bbc_text[bbc_text.category=="sport"]
text=subset.text.values
words =" ".join(text)
create_wordcloud(words)

#Displaying the word cloud for the tech category
subset=bbc_text[bbc_text.category=="tech"]
text=subset.text.values
words =" ".join(text)
create_wordcloud(words)

### As there are a totla of 5 classes of news that we have here in our dataset.
# As our model would require it in numeric form, lets map it to numeric form.

bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})
bbc_text.category.unique()

bbc_text.info()

### Train Test Split


X = bbc_text.text
y = bbc_text.category

#split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 1)
print(X_train)
print(y_train)

#Creating the Bag of Words Representation
# countVectorizer

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words = 'english')
# fit the vectorizer on the training data

vec.fit(X_train)
print(len(vec.get_feature_names()))
vec.vocabulary_

# another way of representing the features
X_transformed = vec.transform(X_train)
X_transformed
print(X_transformed)

X_transformed.toarray()

# convert X_transformed to sparse matrix, just for readability.
pd.DataFrame(X_transformed.toarray(), columns= [vec.get_feature_names()])

# for test data
X_test_transformed = vec.transform(X_test)
X_test_transformed

print(X_test_transformed)

# convert X_transformed to sparse matrix, just for readability
pd.DataFrame(X_test_transformed.toarray(), columns= [vec.get_feature_names()])

### Building the model

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()
logit.fit(X_transformed, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
# fit
logit.fit(X_transformed,y_train)

# predict class
y_pred_class = logit.predict(X_test_transformed)

# predict probabilities
y_pred_proba = logit.predict_proba(X_test_transformed)

#Model Evaluation Logistic Regression
# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)

specificity = TN / float(TN + FP)
print("specificity",specificity)

print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class, average = 'micro'))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class, average = 'micro'))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class, average = 'micro'))

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_transformed, y_train)

# fit
nb.fit(X_transformed,y_train)

# predict class
y_pred_class = nb.predict(X_test_transformed)

# predict probabilities
y_pred_proba = nb.predict_proba(X_test_transformed)

#Model Evaluation Naive Bayes
# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
# confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)

specificity = TN / float(TN + FP)
print("specificity",specificity)

print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class, average = 'micro'))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class, average = 'micro'))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class, average = 'micro'))

#Both the Logistic Regression as well as the Naive Bayes model offer similar performance.
# #We will go ahead choosing Naive bayes as our final model.

abc = ['Facebook has hit out at a ban on its platforms introduced in Russia on Friday amid the ongoing war in Ukraine.']
vec1 = vec.transform(abc).toarray()
print('Text :' ,abc)
print('category : ' ,str(list(nb.predict(vec1))[0]).replace('0', 'TECH').
      replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))