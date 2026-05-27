import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from features import X_scaled, y
from sklearn.preprocessing import (
   LabelEncoder,
   StandardScaler
)
import pandas as pd

encoder=LabelEncoder()
scaler=StandardScaler()
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
####################################

# Count class frequencies
class_counts = pd.Series(y).value_counts()

print(class_counts)

# Keep classes with >=10 samples
valid_classes = class_counts[class_counts >= 10].index

# Filter data
mask = np.isin(y, valid_classes)

X_scaled = X_scaled[mask]
y = y[mask]

print("Filtered Shape:", X_scaled.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Compute class weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class Weights:")
print(class_weights)

# Number of output classes
num_classes = len(np.unique(y))

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(32, activation='relu'),

    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Predictions
y_pred_probs = model.predict(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
model.save("models/cybersecurity_model.keras")

print("\nModel saved successfully")


#import joblib

joblib.dump(encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Preprocessing objects saved successfully")