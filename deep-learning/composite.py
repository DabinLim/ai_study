from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 패딩 : 입력 및 출력의 크기와 커널의 크기가 다른 경우 크기를 맞추기 위해 입력 배열 주위를 가상의 원소로 채우는 것
# 세임패딩: 입력 주위에 0으로 패딩 하는 것
# 밸리드 패딩 : 순수한 입력 배열에서만 합성곱을 하여 특성 맵을 만드는 경우
# 스트라이드 : 패딩을 만들기 위한 이동의 크기 (몇칸씩 이동하는지)
# 첫번째 매개변수는 필터(커널)의 갯수
# keras.layers.Conv2D(
#     10,
#     kernel_size=(3, 3),
#     activation='relu',
#     padding='same',
#     strides=1)

# 풀링 : 합성곱 층에서 만든 특성 맵의 가로세로 크기를 줄이는 역할을 수행, 특성맵의 개수를 줄이지는 않는다
# 예를 들어 (2, 2, 3) 크기의 특성 맵에 풀링을 적용하면 (1, 1, 3) 크기의 특성 맵이 된다.
# 풀링에는 가중치가 없으며 도장을 찍은 영역에서 가장 큰 값을 고르거나 평균값을 계산한다 (최대 풀링, 평균 풀링)
# 풀링 층의 출력 또한 특성 맵이라고 한다.
# 풀링에서는 겹치지 않고 이동하기 때문에 풀링의 크기가 (2, 2)이면 가로세로 두 칸씩 이동한다. 즉, 스트라이드가 2이다.
# 첫번째 매개변수는 풀링의 크기 (가로세로 크기의 크기를 절반으로 줄임) padding 은 valid 로 패딩을 하지 않는다.
# 대부분 최대풀링을 많이 사용한다, 평균풀링은 특성 맵에 있는 중요한 정보를 희석시킬 수 있다
# 평균풀링 클래스는 AveragePooling2D로 제공하는 매개변수는 최대풀링과 같다
# keras.layers.MaxPooling2D(2, strides=2, padding='valid')

(train_input,
 train_target), (test_input,
                 test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)


model = keras.Sequential()
model.add(
    keras.layers.Conv2D(
        32,
        kernel_size=3,
        activation='relu',
        padding='same',
        input_shape=(28, 28, 1)
    )
)
model.add(keras.layers.MaxPooling2D(2))
model.add(
    keras.layers.Conv2D(
        64,
        kernel_size=3,
        activation='relu',
        padding='same',
    )
)
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
print(model.summary())

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(
    val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
