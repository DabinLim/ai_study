import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

pca = PCA(n_components=50)
pca.fit(fruits_2d)

# (300, 10000) 10000개 특성의 300개 데이터
print(fruits_2d.shape)

fruits_pca = pca.transform(fruits_2d)
# (300, 50)
print(fruits_pca.shape)

# 차원 복구
# (300, 10000)
fruits_inverse = pca.inverse_transform(fruits_pca)
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
print(fruits_inverse.shape)


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(
        cols*ratio, rows*ratio), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')  # type: ignore
    plt.show()


lr = LogisticRegression()
# 로지스틱 회귀로 학습
target = np.array([0]*100 + [1]*100 + [2] * 100)
scores = cross_validate(lr, fruits_2d, target)

# 특성이 10000개, 샘플이 300개 이므로 과대적합
# 0.9966666666666667
# 0.48713297843933107
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# pca로 축소된 데이터 훈련
scores_pca = cross_validate(lr, fruits_pca, target)

# 훈련시간 감소
# 1.0
# 0.013804006576538085
print(np.mean(scores_pca['test_score']))
print(np.mean(scores_pca['fit_time']))


pca0_5 = PCA(n_components=0.5)
pca0_5.fit(fruits_2d)

# 2개의 특성
print(pca.n_components_)

fruits_pca0_5 = pca0_5.transform(fruits_2d)

scores_pca0_5 = cross_validate(lr, fruits_pca0_5, target)
# 0.9933333333333334
# 0.0221343994140625
print(np.mean(scores_pca0_5['test_score']))
print(np.mean(scores_pca0_5['fit_time']))

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca0_5)
# for label in range(0, 3):
#     draw_fruits(fruits[km.labels_ == label])

for label in range(0, 3):
    data = fruits_pca0_5[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
