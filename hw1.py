from sklearn.feature_extraction.text import TfidfVectorizer
import json
import glob
from pandas import DataFrame

paths = glob.glob("rus_x/*.txt")
print(paths)
texts = [open(x, 'r').read() for x in paths]  #I know there's a memory leak here but I don't care
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
print(tfidf_matrix)

with open("rus__tfidf.csv", "w") as io:
    io.write(DataFrame(tfidf_matrix.A, columns=vectorizer.get_feature_names()).irow(0).sort_values(ascending=False).to_csv())