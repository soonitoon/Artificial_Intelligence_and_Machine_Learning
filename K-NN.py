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

# feature_name과 target을 레코드로 갖는 데이터프레임 생성
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 숫자로 코딩된 데이터 문자열 매핑
df['target'] = df['target'].map({0: "setosa", 1: "versicolor", 2: "virginica"})
print(df)
