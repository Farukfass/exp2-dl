#code:
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1,28,28,1).astype('float32') / 255
X_test = X_test.reshape(-1,28,28,1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Build model
model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
MaxPooling2D(pool_size=(2,2)),
Flatten(),
Dense(128, activation='relu'),
Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32,
validation_data=(X_test, y_test))

#output with code(screenshot):<img width="520" height="447" alt="Screenshot 2025-08-13 112556" src="https://github.com/user-attachments/assets/fedcdabd-7724-4963-87cf-5c422dc05d5b" />
<img width="1023" height="288" alt="Screenshot 2025-08-13 112619" src="https://github.com/user-attachments/assets/659c5a32-02e5-4eb5-bd48-3dd0565fb32d" />
