
#code:
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(128, activation='relu'),
Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test,
y_test))

#output with codeI(screenshot):
<img width="614" height="381" alt="Screenshot 2025-08-13 112517" src="https://github.com/user-attachments/assets/c84b64f1-5c33-4c0a-b0e7-978b063b929e" />
<img width="944" height="172" alt="Screenshot 2025-08-13 112530" src="https://github.com/user-attachments/assets/22fc14df-dede-4c0b-a8cb-1295da97c6ce" />
<img width="1487" height="519" alt="Screenshot 2025-08-13 114716" src="https://github.com/user-attachments/assets/08ff7b62-b485-4b32-bf4b-57599158a90c" />
