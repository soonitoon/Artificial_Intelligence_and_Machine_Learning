from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 데이터 로드
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target
