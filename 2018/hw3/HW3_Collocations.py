import pandas as pd
from pymorphy2 import MorphAnalyzer
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk import FreqDist
from nltk import bigrams
from nltk.metrics import spearman

analyzer = MorphAnalyzer()
corpus = pd.read_csv("court-V-N.csv", header=None)
measures = BigramAssocMeasures()
tagger = lambda x: (x, analyzer.parse(x.lower().strip())[0].tag.POS)
tagged_corpus = corpus.applymap(tagger).drop(0, axis=1)
with open("gold_standard.txt", "r") as io:
    standard = [tuple(x.split()) for x in io.readlines()]
wfd = FreqDist(tagged_corpus.values.flatten())
bfd = FreqDist(bigrams(tagged_corpus.values.flatten()))
finder_1 = BigramCollocationFinder(wfd, bfd)

filter = lambda x: [tuple(z[0] for z in y[0]) for y in x if y[0][0][1] == "INFN"]

scored_pmi = filter(finder_1.score_ngrams(measures.pmi))
scored_student = filter(finder_1.score_ngrams(measures.student_t))
pmi_top = scored_pmi[:10]
student_top = scored_student[:10]

for name, top in [("pmi_top10.txt", pmi_top), ("student_top10.txt", student_top)]:
    with open(name, "w") as io:
        joined = [" ".join(x) + "\n" for x in top]
        io.writelines(joined)

print(spearman.spearman_correlation(pmi_top, student_top))
print("Done")
