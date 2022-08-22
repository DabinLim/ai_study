from tensorflow import keras
from sklearn.model_selection import cross_validate, train_test_split

(train_input,
 train_target), (test_input,
                 test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
# train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 활성화 함수 : 은닉층에서 산술 계산만 했을 경우 수행 역할이 없어지는 경우를 대비해 선형 계산을 비성현적으로 비틀어 주는 역할
# dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
# dense2 = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential()
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)
rmsprop = keras.optimizers.RMSprop()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
