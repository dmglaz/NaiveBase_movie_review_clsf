from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import NaiveBayesClassifier, classify
from random import choice, shuffle
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.corpus import stopwords, gutenberg, brown, movie_reviews
from os import getcwd
import csv

documents = []
ponctuation = [',', '.', "'", '"', '-', ')', '(', '?', "!", ':']

for category in movie_reviews.categories():
    for doc_name in movie_reviews.fileids(category):
       movie_rev = list(movie_reviews.words(doc_name))
       movie_rev = [w.lower() for w in movie_rev if w not in ponctuation]
       documents.append((movie_rev, category))
shuffle(documents)
# ----------------------------------------------------------------------------
all_words = [w.lower() for w in movie_reviews.words()]
words_freqs = FreqDist(all_words) # words_freqs = {word: number of times the word appeard in the list given to the FreqDist}

common_words = [word for word, freq in words_freqs.most_common(10000) if (word not in stopwords.words("english")) and (word not in ponctuation)]
print(common_words[:100])
# -------Funtions---------------------------------------------------------
def find_features(document, com_words = common_words):
    words = set(document)
    features = {}
    for w in com_words:
        features[w] = (w in words)
    return features
# ---------------------------------------------------------------------------

feature_sets = [(find_features(text), category) for (text, category) in documents]
data = {}
data["train"] = feature_sets[:1900]
data["test"] = feature_sets[1900:]
clf = NaiveBayesClassifier.train(data["train"]) # acc: 85.095
# acc = classify.accuracy(clf, data["test"])*100
clf.show_most_informative_features(10)

#------TEST------------------------------------
rev_name = movie_reviews.fileids("neg")[11]
text = movie_reviews.words(rev_name)
clf.classify(find_features(text))