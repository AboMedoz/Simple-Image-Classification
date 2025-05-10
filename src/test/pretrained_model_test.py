import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
TEST_PATH = os.path.join(ROOT, 'data', 'dataset', 'test_batch')
MODEL_PATH = os.path.join(ROOT, 'model', 'imagenet_model.keras')

data = []
labels = []

with open(TEST_PATH, 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
    data.append(batch[b'data'])
    labels.append(batch[b'labels'])

x = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
y = np.hstack(labels)

model = load_model(MODEL_PATH)

pred = model.predict(x)
pred_class = np.argmax(pred, axis=1)

print(f"Accuracy: {accuracy_score(y, pred_class) * 100:.2f}")
print(f"Classification Report: {classification_report(y, pred_class)}")