import nltk, string, numpy
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import json
data = json.load(open('mlproject/newplots.json'))
titles = [data[d]['title'] for d in range(len(data))]

dist = numpy.loadtxt("dist2.csv", delimiter=",")
for i in range(10):
    linkage_matrix = ward(dist[i*100:(i+1)*100]) #define the linkage_matrix using ward clustering pre-computed distances
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

    plt.tick_params(\

        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    # plt.tight_layout() #show plot with tight layout
    #uncomment below to save figure
    plt.savefig('ward_clusters'+str(i)+'.png', dpi=200) #save figure as ward_clusters
    plt.show()
