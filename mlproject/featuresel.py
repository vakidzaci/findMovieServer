import nltk, string, numpy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json
import numpy as np
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



data = json.load(open('mlproject/newplots.json'))
documents = [data[d]['plot'] for d in range(len(data))]
s = "Tony Stark creates the Ultron Program to protect the world, but when the peacekeeping program becomes hostile, The Avengers go into action to try and defeat a virtually impossible enemy together. Earth's mightiest heroes must come together once again to protect the world from global extinction."
documents.add(s)
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(documents)
tf_matrix = LemVectorizer.transform(documents).toarray()
print tf_matrix.shape
numpy.savetxt("data.csv", tf_matrix, delimiter=",")
