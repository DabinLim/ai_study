from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier

(train_input,
 train_target), (test_input,
                 test_target) = keras.datasets.fashion_mnist.load_data()

# print(train_input.shape, train_target.shape)

# fig, axs = plt.subplots(1, 10, figsize=(10, 10))
# for i in range(10):
#     axs[i].imshow(train_input[i], cmap='gray_r')  # type: ignore
#     axs[i].axis('off')   # type: ignore

# 샘플 갯수 출력
# print(np.unique(train_target, return_counts=True))

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

# # 확률적 경사하강법 + 교차 검증
# sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
# scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
# 0.8197166666666666
# print(np.mean(scores['test_score']))

# 딥러닝의 데이터셋은 충분히 크기 때문에 검증 점수가 안정적이고 교차 검증을 수행하기에는 훈련 시간이 너무 오래 걸리므로 검증 세트를 따로 떼어낸다
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)

# 이진분류 : loss = 'binary_crossentropy'
# 다중분류 : loss = 'categorical_crossentropy'
# sparse_categorical_crossentropy : 원-핫 인코딩으로 바꾸어 손실 계산

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
