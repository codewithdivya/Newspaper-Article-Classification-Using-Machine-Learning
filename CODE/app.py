from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import accuracy_score

webapp = Flask(__name__)


@webapp.route('/')
def index():
    return render_template('index.html')


@webapp.route('/load', methods=["GET", "POST"])
def load():
    global df, dataset
    if request.method == "POST":
        file = request.files['file']
        df = pd.read_csv(file)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')


@webapp.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@webapp.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global X, y, X_train, X_test, y_train, y_test, X_transformed, X_test_transformed, vec
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df.dropna(axis=0, inplace=True)
        df.drop_duplicates(inplace=True)
        df.category = df.category.map({'tech': 0, 'business': 1, 'sport': 2, 'entertainment': 3, 'politics': 4})
        X = df.text
        y = df.category
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
        print(X_train)
        print(y_train)
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(stop_words='english')
        vec.fit(X_train)
        print(len(vec.get_feature_names()))
        vec.vocabulary_
        X_transformed = vec.transform(X_train)
        X_transformed.toarray()
        print(X_train)
        pd.DataFrame(X_transformed.toarray(), columns=[vec.get_feature_names()])
        X_test_transformed = vec.transform(X_test)
        pd.DataFrame(X_test_transformed.toarray(), columns=[vec.get_feature_names()])

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')


@webapp.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            clf = LogisticRegression(random_state=1).fit(X_transformed, y_train)
            model = clf
            pred = clf.predict(X_test_transformed)
            ac_lr = accuracy_score(y_test, pred)
            ac_lr = ac_lr * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Logistic Regression is ' + str(ac_lr) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            nb = MultinomialNB().fit(X_transformed, y_train)
            model = nb
            nb_pred = nb.predict(X_test_transformed)
            ac_nb = accuracy_score(y_test, nb_pred)
            ac_rf = ac_nb * 100
            msg = 'The accuracy obtained by Naive Bayes is ' + str(ac_rf) + str('%')
            return render_template('model.html', msg=msg)

    return render_template('model.html')


@webapp.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # f1=int(request.form['city'])
        f2 = request.form['gggg']
        print(f2)
        print(type(f2))

        # model.fit(X_transformed, y_train)
        f2 = vec.transform([f2]).toarray()
        result = model.predict(f2)
        print(result)
        # ({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})
        if result == 0:
            msg = 'Tech'
            return render_template('prediction.html', msg=msg)
        elif result == 1:
            msg = 'Business'
            return render_template('prediction.html', msg=msg)
        elif result == 2:
            msg = 'Sports'
            return render_template('prediction.html', msg=msg)
        elif result == 3:
            msg = 'Entertainment'
            return render_template('prediction.html', msg=msg)
        else:
            msg = 'Politics'
            return render_template('prediction.html', msg=msg)
    return render_template('prediction.html')


if __name__ == '__main__':
    webapp.run(debug=True)
