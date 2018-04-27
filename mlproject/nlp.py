import nltk, string, numpy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


d1 = "plot: two teen couples go to a church party, drink and then drive."
d2 = "plot: three teen couples go to a church party, eat and then drive."

# d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
# d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
# d4 = "damn that y2k bug . "
documents = [d1,
             d2
             # d3,
             # d4
             ]


LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(documents)
stemmer = nltk.stem.porter.PorterStemmer()

# print LemVectorizer.vocabulary_

tf_matrix = LemVectorizer.transform(documents).toarray()
# print tf_matrix
# print tf_matrix.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)
# print tfidfTran.idf_

import math
def idf(n,df):
    result = math.log((n+1.0)/(df+1.0)) + 1
    return result
# print "The idf for terms that appear in one document: " + str(idf(4,1))
# print "The idf for terms that appear in two documents: " + str(idf(4,2))

tfidf_matrix = tfidfTran.transform(tf_matrix)
# print tfidf_matrix.toarray()

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print cos_similarity_matrix
