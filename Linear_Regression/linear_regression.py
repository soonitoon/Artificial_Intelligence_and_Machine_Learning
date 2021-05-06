import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

X = [
    [174],
    [152],
    [138],
    [128],
    [186]
]
y = [71, 55, 46, 38, 88]
reg.fit(X, y)  # 학습

print(reg.predict([[165]]))

# 학습 데이터와 y 값을 산포도로 그리기
plt.scatter(X, y, color='black')

# 학습 데이터를 입력으로 해서 예측값 계산
y_pred = reg.predict(X)

# 회귀선 그리기
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.show()
