import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
                         21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
                         23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
                         27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
                         39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
                         44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                         115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                         218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                         556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                         850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
                         1000.0])

# plt.scatter(perch_length, perch_weight)
# plt.xlabel('length')
# plt.xlabel('weight')
# plt.show()

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# 2차원 배열로 변경
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# 예측을 위한 회귀
knr = KNeighborsRegressor()

knr.fit(train_input, train_target)


# 테스트 세트에 대한 예측
test_prediction = knr.predict(test_input)

# 결정계수 (R^2) : 테스트 세트에 대한 평균 절댓값 오차
# 회귀 문제의 성능 측정 도구, 1에 가까울수록 좋고 0에 가까울수록 성능이 나쁘다.
mae = mean_absolute_error(test_target, test_prediction)


# 과대 적합
# 훈련세트에서는 점수가 좋았는데 테스트 세트에서 점수가 나쁜 경우

# 과소 적합
# 훈련세트보다 테스트 세트의 점수가 더 높거나 두 점수가 모두 너무 낮은 경우
# 모델이 너무 단순하여 훈련 세트에 적절히 훈련되지 않은 경우

# k-최근접 이웃 알고리즘으로 모델을 복잡하게 만드는 방법은 이웃의 개수 k를 줄이는 것이다.
knr.n_neighbors = 3

knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
