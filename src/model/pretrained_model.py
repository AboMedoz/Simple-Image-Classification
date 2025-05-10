import os
import pickle

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Resizing, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'dataset')  # or 'train_batch'

data = []
labels = []

for i in range(1, 6):
    batch = os.path.join(DATA_PATH, f'data_batch_{i}')
    with open(batch, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data.append(batch[b'data'])
        labels.extend(batch[b'labels'])

x = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
y = np.array(labels)
y_cat = to_categorical(y, num_classes=10)

input_layer = Input(shape=(32, 32, 3))
resized = Resizing(96, 96)(input_layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=resized)
x_out = GlobalAveragePooling2D()(base_model.output)
output = Dense(10, activation='softmax')(x_out)
model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    x, y_cat,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

model.save('imagenet_model.keras')
print("Model saved")

