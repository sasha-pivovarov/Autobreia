import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pymorphy2 import MorphAnalyzer
import re
from string import punctuation
from sklearn.metrics import classification_report, f1_score


context = ("железобетонный|деревянный|металлический" +"|"+ "язык|языковой|лингвистика|эргативный|номинативный|грамматический|частотный").split("|")

flatten = lambda x: [item for sublist in x for item in sublist]
regex = re.compile("[%s0-9]" % (punctuation + "—"))
analyzer = MorphAnalyzer()
pca = PCA(n_components=350)

def preprocess(text:str):
    cleaned = regex.sub("", text.lower())
    words = [analyzer.parse(x)[0].normal_form for x in cleaned.split()]
    words = [x for x in words if x not in context]
    return words

def pad(vector, longest):
    diff = longest - len(vector)
    padding = [0 for x in range(diff)]
    return vector.extend(padding)

def encode(X, vocab):
    onehot = []
    for wordlist in X.values:
        onehot.append([int(voc_item in wordlist) for voc_item in vocab])

    return np.array(onehot)

corpus = pd.read_csv("corpus.csv")
test_corpus = pd.read_csv("test.csv")

X = corpus.apply(lambda x: x["Left context"] + " " + x["Right context"], axis=1)
X = X.apply(lambda x: preprocess(x))
vocab = list(set(flatten(X.values)))
onehot = encode(X, vocab)
X = onehot
X = pca.fit_transform(X)
y = corpus["Class"]
# X = label.fit_transform(X)
print(len(vocab))
kfold = KFold(n_splits=7, shuffle=True, random_state=1337)

fscores = []
for train, test in kfold.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    model = GridSearchCV(BernoulliNB(), {"alpha":[5, 1, 0.8, 0.5], "fit_prior":[True, False]})
    model.fit(X_train, y_train)
    print(model.best_params_)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    fscores.append(f1_score(y_test, y_pred))

# {'alpha': 1, 'fit_prior': True}
#              precision    recall  f1-score   support
#
#           0       0.59      0.69      0.63        29
#           1       0.61      0.50      0.55        28
#
# avg / total       0.60      0.60      0.59        57
#
# {'alpha': 5, 'fit_prior': True}
#              precision    recall  f1-score   support
#
#           0       0.73      0.86      0.79        28
#           1       0.83      0.69      0.75        29
#
# avg / total       0.78      0.77      0.77        57
#
# {'alpha': 5, 'fit_prior': True}
#              precision    recall  f1-score   support
#
#           0       0.64      0.59      0.62        27
#           1       0.66      0.70      0.68        30
#
# avg / total       0.65      0.65      0.65        57
#
# {'alpha': 1, 'fit_prior': False}
#              precision    recall  f1-score   support
#
#           0       0.73      0.64      0.68        25
#           1       0.74      0.81      0.77        31
#
# avg / total       0.73      0.73      0.73        56
#
# {'alpha': 5, 'fit_prior': True}
#              precision    recall  f1-score   support
#
#           0       0.56      0.60      0.58        25
#           1       0.66      0.61      0.63        31
#
# avg / total       0.61      0.61      0.61        56
#
# {'alpha': 5, 'fit_prior': False}
#              precision    recall  f1-score   support
#
#           0       0.59      0.52      0.55        33
#           1       0.41      0.48      0.44        23
#
# avg / total       0.51      0.50      0.50        56
#
# {'alpha': 5, 'fit_prior': False}
#              precision    recall  f1-score   support
#
#           0       0.66      0.61      0.63        31
#           1       0.56      0.60      0.58        25
#
# avg / total       0.61      0.61      0.61        56


print(np.mean(fscores))
# 0.628663303329
tx = test_corpus.apply(lambda x: x["Left context"] + " " + x["Right context"], axis=1)
tx = tx.apply(lambda x: preprocess(x))
tx = encode(tx, vocab)
tx = pca.transform(tx)
ty = test_corpus["Class"]

model = BernoulliNB()
model.fit(X, y)
y_pred = model.predict(tx)
print(y_pred)
print(classification_report(ty, y_pred))

# [1 1 0 0 1 1 1 1]
#              precision    recall  f1-score   support
#
#           0       1.00      0.50      0.67         4
#           1       0.67      1.00      0.80         4
#
# avg / total       0.83      0.75      0.73         8
