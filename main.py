import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA


anime = pd.read_csv('dataset/anime.csv')

anime.head()

rating = pd.read_csv('dataset/rating.csv')
rating.head(10)

rating[rating['user_id']==1].rating.mean()
rating[rating['user_id']==2].rating.mean()

Mean_Rating = rating.groupby(['user_id']).mean().reset_index()
Mean_Rating['mean_rating'] = Mean_Rating['rating']

Mean_Rating.drop(['anime_id','rating'],axis=1, inplace=True)
Mean_Rating.head(10)

rating = pd.merge(rating,Mean_Rating,on=['user_id','user_id'])
rating.head(5)

rating = rating.drop(rating[rating.rating < rating.mean_rating].index)

rating[rating['user_id']== 1].head(10)
rating[rating['user_id']== 8].head(10)

rating["user_id"].unique()

rating = rating.rename({'rating':'userRating'}, axis='columns')

newdf = pd.merge(anime,rating,on=['anime_id','anime_id'])
newdf= newdf[newdf.user_id <= 20000]
newdf.head(10)

len(newdf['anime_id'].unique())

len(anime['anime_id'].unique())

animelist= pd.crosstab(newdf['user_id'], newdf['name'])

pca = PCA(n_components=3)
pca.fit(animelist)
pcaS = pca.transform(animelist)


ps = pd.DataFrame(pcaS)
print(ps.head())

clusterS = pd.DataFrame(ps[[0,1,2]])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = []
inertia_l = np.empty(8)

for i in range(2, 8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(clusterS)
    inertia_l[i] = kmeans.inertia_
    scores.append(silhouette_score(clusterS, kmeans.labels_))

plt.plot(range(0,8),inertia_l,'-o')
plt.xlabel('Number of cluster')
plt.axvline(x=4, color='blue', linestyle='--')
plt.ylabel('Inertia')
plt.show()


from sklearn.cluster import KMeans

clusteredDataset = KMeans(n_clusters=4,random_state=30).fit(clusterS)
centers = clusteredDataset.cluster_centers_
c_preds = clusteredDataset.predict(clusterS)

print(centers)
animelist['cluster'] = c_preds

c0 = animelist[animelist['cluster']==0].drop('cluster',axis=1).mean()
c1 = animelist[animelist['cluster']==1].drop('cluster',axis=1).mean()
c2 = animelist[animelist['cluster']==2].drop('cluster',axis=1).mean()
c3 = animelist[animelist['cluster']==3].drop('cluster',axis=1).mean()

c0.sort_values(ascending=False)[0:15]