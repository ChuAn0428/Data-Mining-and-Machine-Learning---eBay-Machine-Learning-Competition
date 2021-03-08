# -*- coding: utf-8 -*-
"""
@author: Chu-An Tsai
"""
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.cluster import contingency_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


data_label = np.loadtxt('label_12345.csv', encoding='ISO-8859-15', delimiter='\t', dtype=str)
data = np.loadtxt('cat_12345.csv', encoding='ISO-8859-15',delimiter='\n', dtype=str)

###################################### Tf-idf
print('Start computing TF-IDF ...')
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data)

print('The data shape of Tf-idf matrix:')
print(X.shape)

###################################### TSVD
N = 200
print('Start reducing the dimensions to',N,'by using TSVD ...')  
tsvd = TruncatedSVD(n_components=N)
X_TSVD = tsvd.fit(X).transform(X)
print(X_TSVD.shape)

X_TSVD2 = np.column_stack((X_TSVD, data_label))
X_TSVD2 = np.array(X_TSVD2)

New_X_TSVD = []
for i in range(len(X_TSVD2)):
    if X_TSVD2[i][-1] != 'NA':
        New_X_TSVD.append(X_TSVD2[i])
           
np.save('New_X_TSVD_Total.npy',New_X_TSVD)

New_X_TSVD = np.load('New_X_TSVD_Total.npy')
New_X_TSVD1 = New_X_TSVD[:, :-1].astype(float)
np.savetxt('Total_TXVD.csv', New_X_TSVD1)
New_X_TSVD_label = New_X_TSVD[:,-1].astype(float)
np.savetxt('Total_TXVD_label.csv', New_X_TSVD_label)

#New_X_TSVD1 = np.loadtxt('Total_TXVD.csv')
#New_X_TSVD_label = np.loadtxt('Total_TXVD_label.csv')

#label_true = np.unique(New_X_TSVD_label)
#print('cluster number:', len(label_true))


#################################### Rand Index Accuracy
def rand_index(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a +=1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b +=1 
            else:
                pass
    RI = (a + b) / (n*(n-1)/2)
    return RI
#1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
###################################### AgglomerativeClustering (hierarchy)
#for n_cs in range(250,len(New_X_TSVD1),50): cat1:0.2 cat2:0.18 cat3:1.5e-2 cat4:1.5e-3 cat5:1.5e-1 
D_T = [1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11, 1e-10, 1e-9, 1e-8]
print('Category Total:')

print("Start computing Agglomerative Clustering clustering ... D = 0")
AgglomerativeClustering_label = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward').fit_predict(New_X_TSVD1)
print('Baseline   RI:',rand_index(AgglomerativeClustering_label, New_X_TSVD_label))

for DT in D_T:
#DT =  cat1:0.2 cat2:0.18 cat3:1.5e-2 cat4:1e-16 cat5:1.5e-1
#print("Start computing AgglomerativeClustering clustering ... N =",n_cs)
        print("Start computing Agglomerative Clustering clustering ... D =",DT)
        #AgglomerativeClustering_label = AgglomerativeClustering(n_clusters=n_cs,linkage='average').fit_predict(New_X_TSVD1)
        AgglomerativeClustering_label = AgglomerativeClustering(n_clusters=None, distance_threshold=DT, linkage='ward').fit_predict(New_X_TSVD1)
        print('"Ward"     RI:',rand_index(AgglomerativeClustering_label, New_X_TSVD_label))
        AgglomerativeClustering_label = AgglomerativeClustering(n_clusters=None, distance_threshold=DT, linkage='single').fit_predict(New_X_TSVD1)
        print('"Single"   RI:',rand_index(AgglomerativeClustering_label, New_X_TSVD_label))
        AgglomerativeClustering_label = AgglomerativeClustering(n_clusters=None, distance_threshold=DT, linkage='average').fit_predict(New_X_TSVD1)
        print('"Average"  RI:',rand_index(AgglomerativeClustering_label, New_X_TSVD_label))
        AgglomerativeClustering_label = AgglomerativeClustering(n_clusters=None, distance_threshold=DT, linkage='complete').fit_predict(New_X_TSVD1)
        print('"Complete" RI:',rand_index(AgglomerativeClustering_label, New_X_TSVD_label))

#pre_label = np.unique(AgglomerativeClustering_label)

############################ Figure

'''
linkage_matrix=linkage(New_X_TSVD1,'average')
dendrogram(linkage_matrix,color_threshold=None)
plt.title('Hierarchical Clustering Dendrogram (Category 2)')
plt.xlabel('samples')
plt.ylabel('distance')
plt.tight_layout()
plt.show()
'''