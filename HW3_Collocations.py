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

wfd = FreqDist(tagged_corpus.values.flatten())
bfd = FreqDist(bigrams(tagged_corpus.values.flatten()))
finder_1 = BigramCollocationFinder(wfd, bfd)

filter = lambda x: [y for y in x if y[0][0][1] == "INFN"]
scored_pmi = filter(finder_1.score_ngrams(measures.pmi))
scored_student = filter(finder_1.score_ngrams(measures.student_t))
scored_chisquared = filter(finder_1.score_ngrams(measures.chi_sq))



print("Done")
