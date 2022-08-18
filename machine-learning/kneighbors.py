import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 도미의 길이
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
# 도미의 무게
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어의 길이
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3,
                11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# 빙어의 무게
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7,
                10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.scatter(30, 600, marker='^')  # type: ignore
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# fish_data = [[l, w] for l, w in zip(length, weight)]

# fish_target = [1] * 35 + [0] * 14

kn = KNeighborsClassifier()
kn49 = KNeighborsClassifier(n_neighbors=49)

# kn.fit(fish_data, fish_target)
# print(kn.score(fish_data, fish_target))

# print(kn.predict([[30,600]]))

# train_input = fish_data[:35]
# train_target = fish_target[:35]
# test_input = fish_data[35:]
# test_target = fish_target[35:]

# kn.fit(train_input, train_target)
# print(kn.score(test_input, test_target))





# 샘플링 편향 해결을 위해 훈련 데이터와 테스트 데이터를 섞는 작업

# input_arr = np.array(fish_data)
# target_arr = np.array(fish_target)

# seed() : 난수 생성시 토대가 될 정수 전달
# np.random.seed(42)
# index = np.arange(49)
# np.random.shuffle(index)

# print(index[:35])
# train_input = input_arr[index[:35]]
# train_target = target_arr[index[:35]]
# test_input = input_arr[index[35:]]
# test_target = target_arr[index[35:]]

# kn.fit(train_input, train_target)

# print(kn.score(test_input, test_target))




# 데이터가 큰 경우 넘파이 배열을 사용하는 것이 효율적

fish_data = np.column_stack((length, weight))


fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# train_test_split() : 훈련 데이터를 훈련 세트와 테스트 세트로 나누는 함수
# default 비율은 0.25, stratify 매개변수에 타깃 데이터를 전달하면 비율에 맞게 나누어 준다.
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)

kn.fit(train_input, train_target)





# 데이터 전처리 (특성의 스케일이 달라 생기는 오류 해결)

# 평균
mean = np.mean(train_input, axis=0)
# 표준편차
std = np.std(train_input, axis=0)


# 표준점수 ((원본데이터 - 평균) / 표준편차) : 훈련 세트의 스케일을 바꾸는 대표적인 방법 중 하나이다.
# train_input의 모든 행에서 mean의 두 평균값을 빼준다. (브로드캐스팅)
train_scaled = (train_input - mean) / std

new = ([25, 150] - mean) / std
# plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
# plt.scatter(new[0], new[1], marker='^')  # type: ignore
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

test_scaled = (test_input - mean) / std

kn.fit(train_scaled, train_target)
print(kn.score(test_scaled, test_target))

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')  # type: ignore
plt.scatter(train_scaled[indexes, 0],
            train_scaled[indexes, 1], marker='D')  # type: ignore
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
