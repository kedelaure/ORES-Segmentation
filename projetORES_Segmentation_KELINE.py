#Projet ORES Segmentation Keline

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
# %matplotlib inline

import seaborn as sns
pd.set_option("display.max_columns",None)

"""### Lecture des données

"""

data = pd.read_csv('data_menage.csv')

data.drop_duplicates(subset=['CD_SECTOR'],inplace=True)
data.shape


df = pd.read_excel('TF_PSNL_INC_TAX_SECTOR.xlsx',
   sheet_name='TF_PSNL_INC_TAX_SECTOR',
    )

df.drop_duplicates(subset=['CD_SECTOR'],inplace=True)
df.shape

df.shape

df['MS_AVG_TOT_NET_TAXABLE_INC'].replace("",np.NaN,inplace=True)

# traitement des données manquantes
mean=df['MS_AVG_TOT_NET_TAXABLE_INC'].mean()
df['MS_AVG_TOT_NET_TAXABLE_INC'].fillna(mean,inplace=True)

df.head()

df['MS_AVG_TOT_NET_TAXABLE_INC'].isnull().sum()

#Renomons la colonne average income

df.rename(columns={'MS_AVG_TOT_NET_TAXABLE_INC':'Avg_income'},inplace=True)
df.head()

df['CD_SECTOR'].isna().sum()

#merge by CD-SECTOR

d=pd.merge(data, df, on='CD_SECTOR',  how='inner')
d.head()

d.shape

dat = pd.read_excel(
   'statbel_sector_fiscal_2017_V3_20200613_YPE.xlsx',
   sheet_name='Segment_David',
    )

#data.head()
data.shape

dat.head()
#df.shape
#df['CD_SECTOR'].unique().shape

dat.shape

#merge final
da=pd.merge(d, dat, on='CD_SECTOR',  how='inner')
#da.head()
#da.tail()
#dat.tail()

da.shape
#dat.shape

da.head()

#da=da.dropna(subset=['POPULATION','DT_STRT_SECTOR','DT_STOP_SECTOR','OPPERVLAKKTE IN HM²','TX_DESCR_SECTOR_NL_y','CD_REFNIS_y','TX_DESCR_SECTOR_FR_y','TX_DESCR_NL_y','TX_DESCR_FR_y','Wal?','density','Avg_income_y','Segment'])

#df = pd.read_csv('dataNew.csv')

#data_2 = pd.merge(data, df, on='CD_SECTOR',  how='left')
data_2=da

data_2.head()

data_2.dropna(axis = 0, subset=['Segment'],inplace=True)

data_2['Avg_income']=data_2['Avg_income_x']


data_interest = data_2[['Moy pers par menages','density','Avg_income']]
data_interest.index = data_2.CD_SECTOR
data_interest.head()

# Transformation en utilisant le logarithme

data_interest['density']=np.log(data_interest['density']+0.001)
data_interest['Avg_income']=np.log(data_interest['Avg_income']+0.001)

data_interest.head()   # pour vérifier que tout est Ok


"""### clustering based on density and avg_income"""

from sklearn.preprocessing import StandardScaler   #ceci permet de standardiser les données
X = data_interest.values[:,1:]
X = np.nan_to_num(X)
# Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet=X


wccs = []
for i in range(1,11):
  k_means = KMeans(init = "k-means++", n_clusters = i, n_init = 12,random_state=10)
  k_means.fit(Clus_dataSet)
  wccs.append(k_means.inertia_)

plt.plot(range(1,11),wccs)
plt.title("optimal number of clusters")
plt.xlabel(" number of clusters")
plt.ylabel("inertia")
plt.show()
plt.savefig("elbow technique 2 features")

"""### Exploring optimal clusters number through silhouette scoring"""

from sklearn.metrics import  silhouette_samples, silhouette_score

for i,k in enumerate(range(3,9)):
  fig,ax = plt.subplots(1,2,figsize = (15,5))

  k_means = KMeans(init = "k-means++", n_clusters = k, n_init = 12,random_state=10)
  k_means.fit(Clus_dataSet)
  y_predict = k_means.labels_
  centroids = k_means.cluster_centers_
  data_interest['Test_cluster'] = y_predict
  centroids_df = data_interest.groupby('Test_cluster')['density','Avg_income'].mean()
  centroids_density = list(centroids_df['density'])
  centroids_income = list(centroids_df['Avg_income'])


  silhouette_vals = silhouette_samples(Clus_dataSet,y_predict)

  y_ticks = []
  y_lower = y_upper = 0
  for i,cluster in enumerate(np.unique(y_predict)):
    cluster_silhouette_vals = silhouette_vals[y_predict ==cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    ax[0].barh(range(y_lower,y_upper),cluster_silhouette_vals,height=1)
    ax[0].text(-0.03,(y_lower+y_upper)/2,str(i+1))
    y_lower += len(cluster_silhouette_vals)

  avg_score = np.mean(silhouette_vals)
  ax[0].axvline(avg_score, linestyle='--',linewidth=2,color='green')
  ax[0].set_yticks([])
  ax[0].set_xlim([-0.1,1])
  ax[0].set_xlabel('Silhouette coefficient values')
  ax[0].set_ylabel('Cluster labels')
  ax[0].set_title('Silhouette plot for the various clusters')

  ax[1].scatter(data_interest['density'],data_interest['Avg_income'],c=y_predict)
  ax[1].scatter(centroids_density,centroids_income,marker = '*',c='r',s=250)
  ax[1].set_xlabel('density')
  ax[1].set_ylabel('Average income')
  ax[1].set_title('Visualization of clustered data',y = 1.02)

  plt.tight_layout()
  plt.suptitle(f'Silhouette analysis using k = {k}, score = {avg_score}',fontsize = 16,fontweight='semibold')
  plt.savefig(f'Silhouettte_analysis_{k}.jpg')
  plt.show

data_interest.head()


clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(Clus_dataSet)
labels = k_means.labels_
print(labels)

k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

data_interest['cluster'] = labels
data_interest.head()

sns.set_style('darkgrid')
sns.set_context( 'paper',font_scale=1.5)
plt.style.use('ggplot')

#plt.figure(figsize=(15,15))

#ax = sns.scatterplot(data=data_interest.iloc[1:1000],x='density',y = 'Avg_income',hue='cluster'
#                  ,palette = 'Set1')
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#plt.show()

centroids = data_2.groupby('Segment')['density','Avg_income'].mean()

centroids

initial_centers = []

for i,row in centroids.iterrows():
  center = [row['density'],row['Avg_income']]
  initial_centers.append(center)
initial_centers

center = np.asarray(initial_centers)
center.shape

#clusterNum = 6
#k_means = KMeans(init = center, n_clusters = clusterNum, n_init = 12)
#k_means.fit(X)
#labels = k_means.labels_

#data_interest['cluster_6'] = labels
#data_interest.head()

#plt.figure(figsize=(8,8))

#ax = sns.scatterplot(data=data_interest.iloc[1:1000],x='density',y = 'Avg_income',hue='cluster_6'
#                 ,palette = 'Set1')
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#plt.show()

"""### clustering based on density, avg_income and avg family members"""

data_interest.head()

data_interest.groupby('Test_cluster')['Test_cluster'].count()

X = data_interest.values[:,0:3]
X

X = data_interest.values[:,0:3]
X = np.nan_to_num(X)
# Clus_dataSet = StandardScaler().fit_transform(X)


Clus_dataSet=X


wccs = []
for i in range(1,11):
  k_means = KMeans(init = "k-means++", n_clusters = i, n_init = 12,random_state=10)
  k_means.fit(Clus_dataSet)
  wccs.append(k_means.inertia_)
wccs

plt.plot(range(1,11),wccs)
plt.title("optimal number of clusters")
plt.xlabel(" number of clusters")
plt.ylabel("inertia")
plt.savefig("elbow technique 3 features")
plt.show()

clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(Clus_dataSet)
labels = k_means.labels_

data_interest['cluster_3D'] = labels
data_interest.head()

X.shape

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('density')
ax.set_ylabel('Avg_income')
ax.set_zlabel('Moyenne famille')
ax.set_title("kmeans clustering with 3 features")
ax.legend(labels = [f'Cluster{i+1}' for i in range(0,4)])
ax.scatter(X[:, 1], X[:, 2], X[:, 0], c= labels.astype(np.float))
plt.savefig("kmeans clustering 3 features")

clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(Clus_dataSet)
labels = k_means.labels_

data_interest['cluster_3D_2'] = labels
data_interest.head()

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('density')
ax.set_ylabel('Avg_income')
ax.set_zlabel('Moyenne famille')
ax.set_title("kmeans clustering with 3 features")
ax.legend(labels = [f'Cluster{i+1}' for i in range(0,4)])
ax.scatter(X[:, 1], X[:, 2], X[:, 0], c= labels.astype(np.float))
plt.savefig("kmeans clustering 3 features 4 clusters 3D")



d_centroids_df = data_interest.groupby('cluster_3D')['density','Avg_income','Moy pers par menages'].mean()
d_centroids_density = list(d_centroids_df['density'])
d_centroids_income = list(d_centroids_df['Avg_income'])
d_centroids_menages = list(d_centroids_df['Moy pers par menages'])
d_centroids_df

d2_centroids_df = data_interest.groupby('cluster_3D_2')['density','Avg_income','Moy pers par menages'].mean()
d2_centroids_density = list(d_centroids_df['density'])
d2_centroids_income = list(d_centroids_df['Avg_income'])
d2_centroids_menages = list(d_centroids_df['Moy pers par menages'])
d2_centroids_df

fig2,axes = plt.subplots(1,3,figsize = (16,8))


axes[0] = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='cluster_3D_2'
                  ,palette = 'Set1',ax = axes[0])
axes[0].scatter(d2_centroids_density,d2_centroids_income,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[0].set_title('clusters in density - income view',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[0].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])

axes[1] = sns.scatterplot(data=data_interest,x='density',y = 'Moy pers par menages',hue='cluster_3D_2'
                  ,palette = 'Set1',ax = axes[1])
axes[1].scatter(d2_centroids_density,d2_centroids_menages,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[1].set_title('clusters in density - household size view',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[1].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))

axes[2] = sns.scatterplot(data=data_interest,x='Avg_income',y = 'Moy pers par menages',hue='cluster_3D_2'
                  ,palette = 'Set1',ax = axes[2])
axes[2].scatter(d2_centroids_income,d2_centroids_menages,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[2].set_title('clusters in income - household size view',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[2].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))



plt.tight_layout()
plt.suptitle(" \n Clustering with 3 features",fontsize = 16,fontweight='semibold')
plt.savefig("3 features kmeans on 3 views")
plt.show()



#plt.figure(figsize=(8,8))

#ax = sns.scatterplot(data=data_2.iloc[1:1000],x='density',y = 'Avg_income',hue='Segment'
#                  ,palette = 'Set1')
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#plt.show()



"""### visualizing a number of 4 clusters and comparing"""

data_interest.head()

X = data_interest.values[:,1:3]
X = np.nan_to_num(X)
# Clus_dataSet = StandardScaler().fit_transform(X)



Clus_dataSet=X



fig,ax = plt.subplots(1,figsize = (8,8))
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12,random_state=10)
k_means.fit(Clus_dataSet)
y_predict = k_means.labels_
centroids = k_means.cluster_centers_
data_interest['Test_cluster'] = y_predict
centroids_df = data_interest.groupby('Test_cluster')['density','Avg_income'].mean()
centroids_density = list(centroids_df['density'])
centroids_income = list(centroids_df['Avg_income'])
ax = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='Test_cluster'
                  ,palette = 'Set1')

#ax.scatter(data_interest['density'],data_interest['Avg_income'],c=y_predict,s=10)
ax.scatter(centroids_density,centroids_income,marker = '^',c='black',s=150)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
ax.set_title('Visualization of clustered data',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig("5 clusters")
plt.show()

"""#### clusters centers"""

centroids_df

data_interest.head()

data_interest.to_csv("data_interest.csv")

fig2,axes = plt.subplots(1,2,figsize = (16,8))


axes[0] = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='Test_cluster'
                  ,palette = 'Set1',ax = axes[0])
axes[0].scatter(centroids_density,centroids_income,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[0].set_title('Visualization of clustered data with 4 clusters',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[0].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
axes[1] =sns.scatterplot(data=data_2,x='density',y = 'Avg_income',hue='Segment'
                  ,palette = 'Set1',ax=axes[1])
#ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[1].set_title('Visualization of the old clusterings with 6 clusters')
old_centroids_df = data_2.groupby('Segment')['density','Avg_income'].mean()
old_centroids_density = list(old_centroids_df['density'])
old_centroids_income = list(old_centroids_df['Avg_income'])
axes[1].scatter(old_centroids_density,old_centroids_income,marker = '^',c='black',s=100)
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.suptitle(" \n Comparaison des clusters",fontsize = 16,fontweight='semibold')
plt.savefig("4 vs 6(old) clusters kmeans")
plt.show()

data_interest.groupby('Test_cluster').count().reset_index()



"""### Hierarchical clustering"""

from sklearn.metrics.pairwise import euclidean_distances
dist_matrix = euclidean_distances(Clus_dataSet,Clus_dataSet) 
print(dist_matrix)

from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering

import pylab
#Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')
#Zw_using_dist_matrix = hierarchy.linkage(dist_matrix, 'ward')

fig = pylab.figure(figsize=(18,50))
#def llf(id):
#    return data_interest.loc[id]
    
#dendro = hierarchy.dendrogram(Z_using_dist_matrix, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'complete')
agglom.fit(dist_matrix)

agglom2 = AgglomerativeClustering(n_clusters = 4, linkage = 'ward')
agglom2.fit(dist_matrix)

#agglom.labels_

data_interest['cluster_hierarchical'] = agglom.labels_
data_interest['cluster_hierarchical_ward'] = agglom2.labels_
data_interest.head()

data_interest['cluster_hierarchical'].unique()

fig,ax = plt.subplots(1,figsize = (8,8))
dendo_centroids_df = data_interest.groupby('cluster_hierarchical')['density','Avg_income'].mean()
dendo_centroids_density = list(centroids_df['density'])
dendo_centroids_income = list(centroids_df['Avg_income'])
ax = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='cluster_hierarchical'
                  ,palette = 'Set1')

#ax.scatter(data_interest['density'],data_interest['Avg_income'],c=y_predict,s=10)
ax.scatter(dendo_centroids_density,dendo_centroids_income,marker = '^',c='black',s=150)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
ax.set_title('Visualization of clustered data',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

fig2,axes = plt.subplots(1,2,figsize = (16,8))


axes[0] = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='Test_cluster'
                  ,palette = 'Set1',ax = axes[0])
axes[0].scatter(centroids_density,centroids_income,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[0].set_title('kmeans clustering',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[0].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
axes[1] =sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='cluster_hierarchical'
                  ,palette = 'Set1',ax=axes[1])
#ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[1].set_title('hierarchical clustering', y = 1.02)
dendo_centroids_df = data_interest.groupby('cluster_hierarchical')['density','Avg_income'].mean()
dendo_centroids_density = list(dendo_centroids_df['density'])
dendo_centroids_income = list(dendo_centroids_df['Avg_income'])
axes[1].scatter(dendo_centroids_density,dendo_centroids_income,marker = '^',c='black',s=100)
axes[1].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.suptitle(" \n Comparing kmeans and hierarchical clustering (complete)",fontsize = 16,fontweight='semibold')
plt.savefig("complete vs kmeans")

from sklearn.cluster import DBSCAN 
epsilon = 0.1
minimumSamples = 10
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(Clus_dataSet)
labels_scan = db.labels_
labels_scan

n_clusters_ = len(set(labels_scan)) - (1 if -1 in labels else 0)
n_clusters_

set(labels_scan)

data_interest['clusters_db_scan'] = labels_scan

data_interest.head()

fig2,axes = plt.subplots(1,2,figsize = (16,8))


axes[0] = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='Test_cluster'
                  ,palette = 'Set1',ax = axes[0])
axes[0].scatter(centroids_density,centroids_income,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[0].set_title('kmeans clustering',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[0].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
data_to_plot = data_interest[data_interest['clusters_db_scan'] != -1]
axes[1] =sns.scatterplot(data=data_to_plot,x='density',y = 'Avg_income',hue='clusters_db_scan'
                  ,palette = 'Set1',ax=axes[1])
#ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[1].set_title('DB scan')
db_centroids_df = data_to_plot.groupby('clusters_db_scan')['density','Avg_income'].mean()
db_centroids_density = list(db_centroids_df['density'])
db_centroids_income = list(db_centroids_df['Avg_income'])
axes[1].scatter(db_centroids_density,db_centroids_income,marker = '^',c='black',s=100)
axes[1].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.suptitle(" Comparing kmeans and DB scan",fontsize = 16,fontweight='semibold')

fig2,axes = plt.subplots(1,2,figsize = (16,8))


axes[0] = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='cluster_hierarchical_ward'
                  ,palette = 'Set1',ax = axes[0])
ward_dendo_centroids_df = data_interest.groupby('cluster_hierarchical_ward')['density','Avg_income'].mean()
ward_dendo_centroids_density = list(ward_dendo_centroids_df['density'])
ward_dendo_centroids_income = list(ward_dendo_centroids_df['Avg_income'])
axes[0].scatter(ward_dendo_centroids_density,ward_dendo_centroids_income,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[0].set_title('hierarchical clustering with ward linkage',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[0].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
axes[1] =sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='cluster_hierarchical'
                  ,palette = 'Set1',ax=axes[1])
#ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[1].set_title('hierarchical clustering with complete linkage',y=1.02)
dendo_centroids_df = data_interest.groupby('cluster_hierarchical')['density','Avg_income'].mean()
dendo_centroids_density = list(dendo_centroids_df['density'])
dendo_centroids_income = list(dendo_centroids_df['Avg_income'])
axes[1].scatter(dendo_centroids_density,dendo_centroids_income,marker = '^',c='black',s=100)
new_labels = [f'Cluster {i+1}' for i in range(0,4)]
axes[1].legend(labels = new_labels)
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.suptitle(" \nComparing ward and complete linkage in hierarchical clustering",fontsize = 16,fontweight='semibold')
plt.savefig("ward vs complete linkage")

fig2,axes = plt.subplots(1,2,figsize = (16,8))


axes[0] = sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='cluster_hierarchical_ward'
                  ,palette = 'Set1',ax = axes[0])
ward_dendo_centroids_df = data_interest.groupby('cluster_hierarchical_ward')['density','Avg_income'].mean()
ward_dendo_centroids_density = list(ward_dendo_centroids_df['density'])
ward_dendo_centroids_income = list(ward_dendo_centroids_df['Avg_income'])
axes[0].scatter(ward_dendo_centroids_density,ward_dendo_centroids_income,marker = '^',c='black',s=100)
#ax.set_xlabel('density')
#ax.set_ylabel('Average income')
axes[0].set_title('hierarchical clustering with ward linkage',y = 1.02)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[0].legend(labels = [f'Cluster {i+1}' for i in range(0,4)])
axes[1] =sns.scatterplot(data=data_interest,x='density',y = 'Avg_income',hue='Test_cluster'
                  ,palette = 'Set1',ax=axes[1])
#ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
axes[1].set_title('kmeans clustering',y=1.02)
#dendo_centroids_df = data_interest.groupby('cluster_hierarchical')['density','Avg_income'].mean()
#dendo_centroids_density = list(dendo_centroids_df['density'])
#dendo_centroids_income = list(dendo_centroids_df['Avg_income'])
axes[1].scatter(centroids_density,centroids_income,marker = '^',c='black',s=100)
new_labels = [f'Cluster {i+1}' for i in range(0,4)]
axes[1].legend(labels = new_labels)
#plt.legend(labels = [i for i in range(0,5)],loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.suptitle(" \nComparing ward linkage (hierarchichal) with kmeans",fontsize = 16,fontweight='semibold')
plt.savefig("ward linkage vs kmeans")

