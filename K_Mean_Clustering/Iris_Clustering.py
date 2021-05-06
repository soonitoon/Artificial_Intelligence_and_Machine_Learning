from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 데이터 로드
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow')
plt.xlabel('Speal Length', fontsize=18)
plt.ylabel('Speal Wength', fontsize=18)

# K-mean 클러스터링
km = KMeans(n_clusters=3, n_jobs=4, random_state=21)
km.fit(X)

# 중심점 위치
centers = km.cluster_centers_
print(centers)

for center in centers:
    plt.plot(center[0], center[1], 'go')

plt.savefig('output_img/k-mean.png')
