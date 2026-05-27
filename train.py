from sklearn.model_selection import train_test_split
from features import X_scaled, y
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#
from tensorflow.keras.callbacks import EarlyStopping
#
from sklearn.preprocessing import (
   LabelEncoder,
   StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import SMOTE
import joblib
import shap


encoder=LabelEncoder()
scaler=StandardScaler()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

######################################################
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
#######################===========+++
#wieghts for imbalanced data
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print(class_weights)
#################################=============
num_classes = len(np.unique(y))

##################################################
model = Sequential()

model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#############################################################################
#############################################################################
#from tensorflow.keras.callbacks import EarlyStopping-OVERFITING

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    class_weight=class_weights,
    callbacks=[early_stop]
)
#EVALUATE
X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)
###############---------------------############
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)   

# Predictions
y_pred = model.predict(X_test)
#PROBABILITY TO LABEL--------ETX
y_pred = np.argmax(y_pred, axis=1)
y_pred = model.predict(X_test)

# Convert probabilities to 0 or 1------ETX
y_pred = (y_pred > 0.5).astype(int)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
print(confusion_matrix(y_test, y_pred))
##########################################################
#etx===================================
mapping = dict(zip(
    encoder.classes_,
    encoder.transform(encoder.classes_)
))
#====================
print(mapping)

model.save("cybersecurity_lstm_model.h5")
#import joblib

joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")


explainer = shap.Explainer(model)
shap_values = explainer(X_test[:100])