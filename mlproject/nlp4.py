import nltk, string, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json

class Movie:
    def __init__(self,_id,title,cos):
        self._id = _id
        self.title = title
        self.cos = cos

    def __gt__(self,another):
        return self.cos > another.cos
    def __repr__(self):
        return str(self._id) + ": "+ self.title + " " +str(float(self.cos))
    def __str__(self):
        return str(self._id) + ": "+ self.title + " " +str(float(self.cos))
    def getJSON(self):
        movie = {
            '_id':self._id,
            'title':self.title,
            'distance':self.cos
        }
        return movie
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


def cos_similarity(textlist,TfidfVec):
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()

def findMovie(plot):

    movies = []
    title = ""
