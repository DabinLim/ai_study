from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(train_input,
 train_target), (test_input,
                 test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)


def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


# model = model_fn()
# print(model.summary())
# # adam은 적응적 학습률을 사용하기 때문에 에포크가 진행되면서 학습률의 크기를 조정할 수 있다.
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics='accuracy')
# # verbose : 훈련과정
# history = model.fit(train_scaled,
#                     train_target,
#                     epochs=20,
#                     # verbose=0,
#                     validation_data=(val_scaled, val_target))

# 인공 신경망 모델이 최적화 하는 대상은 정확도가 아니라 손실 함수이다.
# 모델이 잘 훈련되었는지 파악하려면 손실 함수의 값을 확인하는 것이 더 낫다.

# 드롭아웃 : 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 껴서 과대적합을 막는다.
# 평가와 예측에 모델을 사용할 때는 드롭아웃이 적용되지 않는다.
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')
history = model.fit(train_scaled,
                    train_target,
                    epochs=20,
                    validation_data=(val_scaled, val_target))
