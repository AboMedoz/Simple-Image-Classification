import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, '..')
DATA_PATH = os.path.join(ROOT, 'data', 'dataset')

data = []
labels = []

for i in range(1, 6):
    batch = os.path.join(DATA_PATH, f"data_batch_{i}")
    with open(batch, 'rb') as f:
        data_batch = pickle.load(f, encoding='bytes')
        data.append(data_batch[b'data'])
        labels.append(data_batch[b'labels'])

x = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
y = np.hstack(labels)
y = to_categorical(y, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(datagen.flow(x_train, y_train), epochs=50, batch_size=64, validation_data=(x_test, y_test))

_, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}")

model.save('model.keras')
print('Model Saved')
