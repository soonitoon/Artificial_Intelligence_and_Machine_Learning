from sklearn.datasets import load_iris  # 샘플
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()

# 훈련용, 테스트용으로 각각 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=0)

# 분류
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 예측
y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(y_test, y_pred)
print(scores)

# 임의의 데이터 예측
classes = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
x_new = [
    [3, 4, 6, 1],
    [5, 4, 2, 2]
]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])
