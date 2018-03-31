from pattern.web import Wikipedia
import re
from string import punctuation
from pymorphy2 import MorphAnalyzer
import pandas as pd

safe = lambda x: 0 if x < 0 else x


wiki = Wikipedia(language="ru")
punc = re.compile('[%s]' % re.escape(punctuation))
start_points = ["Проектирование", "Номинативная_конструкция"]
lemma = "конструкция"
translator = str.maketrans('', '', punctuation)
regex = re.compile("[%s0-9]" % (punctuation + "—"))
analyzer = MorphAnalyzer()
stoplinks = ["Википедия", "Английский"]
with open("stop-words-russian.txt") as io:
    stopwords = [x.strip() for x in io.readlines()]


def get_articles(max_count):
    sents = {0: set(), 1: set()}
    for i in range(len(start_points)):
        start = start_points[i]
        links = [start]
        list_of_strings = []
        while len(sents[i]) <= max_count:
            links_temp = []
            for link in links:
                if len(list_of_strings) <= max_count:
                    try:
                        article = wiki.article(link)
                        words = preprocess(article.plaintext())
                        lemma_indices = [i for i, x in enumerate(words) if x==lemma]
                        contexts = [(tuple(words[safe(i-10):i]), tuple(words[i:i+10])) for i in lemma_indices]
                        sents[i].update(contexts)
                        new_links = [x for x in article.links if not any([x.startswith(y) for y in stoplinks])]
                        links_temp.extend(new_links)
                        print(f"Processed link {link}, {len(contexts)} sents found")
                        print(f"Total sent count: {len(sents[i])} out of {max_count}")
                    except AttributeError:
                        print(f"Skipped link {link}")
                        continue
                else:
                    break
            links = links_temp
    return sents


def preprocess(text:str):
    cleaned = regex.sub("", text.lower())
    words = [analyzer.parse(x)[0].normal_form for x in cleaned.split()]
    words = [x for x in words if x not in stopwords]
    return words


sentences = get_articles(160)
for key, value in sentences:
    df = pd.DataFrame(list(sentences))
    df.to_csv(f"{str(key)}.csv")
print("Done for now")

