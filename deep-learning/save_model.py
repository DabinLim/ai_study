from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

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


model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')
# history = model.fit(train_scaled,
#                     train_target,
#                     epochs=10,
#                     validation_data=(val_scaled, val_target))

# model.save_weights('model-weights.h5')
# model.save('model-whole.h5')

# val_labels = np.argmax(model.predict(val_scaled), axis=-1)

# save_best_only : 최상의 검증점수를 저장
# patience : 검증점수가 2회 연속 상승하지 않으면 조기종료
# restore_best_weights : 조기종료시 검증점수가 최상이었던 모델 파라미터로 되돌림
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=2, restore_best_weights=True)
history = model.fit(train_scaled,
                    train_target,
                    epochs=20,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

print(early_stopping_cb.stopped_epoch)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train, val'])
plt.show()
