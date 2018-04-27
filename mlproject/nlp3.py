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
    data = json.load(open('mlproject/newplots.json'))
    movies = []
    title = ""
    for i in range(len(data)):
        documents = [plot,data[i]['plot']]
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        cos = cos_similarity(documents,TfidfVec)[0][1]
        movies.append(Movie(i,data[i]['title'],float(cos)))

    movies.sort()
    print(max(movies))
    movies = movies[::-1]
    movies = movies[:10]
    return [x.getJSON() for x in movies]
d2 = "Captain Jack Sparrow searches for the trident of Poseidon while being pursued by an undead sea captain and his crew."

for i in findMovie(d2):
    print i
