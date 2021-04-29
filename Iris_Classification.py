from sklearn.datasets import load_iris  # 샘플
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=0)

# 분류
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 예측
y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(y_test, y_pred)
print(scores)
