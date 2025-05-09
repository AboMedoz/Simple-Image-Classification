import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODEL_PATH = os.path.join(ROOT, 'model', 'model.keras')
TEST_DATA = os.path.join(ROOT, 'data', 'Dataset', 'test_batch')

data = []
labels = []

with open(TEST_DATA, 'rb') as f:
    data_batch = pickle.load(f, encoding='bytes')
    data.append(data_batch[b'data'])
    labels.append(data_batch[b'labels'])


x = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
y = np.hstack(labels)

model = load_model(MODEL_PATH)

pred = model.predict(x)
pred_classes = np.argmax(pred, axis=1)

print(f"Accuracy: {accuracy_score(y, pred_classes) * 100:.2f}")
print(f"Classification Report: {classification_report(y, pred_classes)}")
