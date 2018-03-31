import pandas as pd
from sklearn.svm import SVC
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

print(np.mean(fscores))
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

