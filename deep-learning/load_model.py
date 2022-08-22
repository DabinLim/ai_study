from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input,
 train_target), (test_input,
                 test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)


model = keras.models.load_model('model-whole.h5')
print(model.evaluate(val_scaled, val_target))
