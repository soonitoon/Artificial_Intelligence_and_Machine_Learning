from sklearn.datasets import load_iris  # 샘플 데이터
import pandas as pd
import numpy as np

# 시각화 패키지
import matplotlib.pyplot as plt
import seaborn as sns

# 샘플 데이터 로드
iris = load_iris()

# key value 확인
print(iris)
print(iris.DESCR)

# label
print(iris.data)
print(iris.feature_names)
