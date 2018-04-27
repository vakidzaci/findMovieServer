import nltk, string, numpy
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import json
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
data = json.load(open('mlproject/plots.json'))
l = len(data)

documents = [data[d]['plot'] for d in range(l)]
titles = [data[d]['title'] for d in range(l)]

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

dist = cosine_similarity(tfidf_matrix)
numpy.savetxt("dist2.csv", dist, delimiter=",")



#linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
#fig, ax = plt.subplots(figsize=(15, 130)) # set size
#ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

#plt.tick_params(\
#    axis= 'x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off')

#plt.tight_layout() #show plot with tight layout
#uncomment below to save figure
#plt.savefig('ward_clusters.png', dpi=500) #save figure as ward_clusters
#plt.show()
