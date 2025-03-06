import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import joblib

# ---------------- STEP 1: LOAD TRAINING ECG DATA ----------------
print("Loading ECG training dataset...")

train_file_path = ""  # Update with your training dataset file
df_train = pd.read_csv(train_file_path, header=None)

# Extract training signals and labels
X_train = df_train.iloc[:, :-1].values  # First 187 columns (ECG signals)
y_train = df_train.iloc[:, -1].values   # Last column (Class labels)

# ---------------- STEP 2: LOAD TEST ECG DATA ----------------
print("Loading ECG test dataset...")

test_file_path = ""  # Update with your test dataset file
df_test = pd.read_csv(test_file_path, header=None)

# Extract test signals and labels
X_test = df_test.iloc[:, :-1].values  # First 187 columns (ECG signals)
y_test = df_test.iloc[:, -1].values   # Last column (Class labels)

# ---------------- STEP 3: PREPROCESS DATA ----------------
# One-hot encode labels (5 classes)
y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# Normalize ECG signals using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)  # Transform test data using same scaler

# Reshape for CNN input (samples, time steps, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ---------------- STEP 4: BUILD CNN MODEL ----------------
print("Building CNN model...")

model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------- STEP 5: TRAIN CNN WITH CHECKPOINT ----------------
print("Training CNN with ModelCheckpoint...")

checkpoint = ModelCheckpoint(
    "best_ecg_cnn_model.h5", monitor='accuracy', verbose=1, 
    save_best_only=True, mode='max'
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint]
)

# ---------------- STEP 6: LOAD BEST MODEL AND TEST ----------------
print("Loading best saved model...")
best_model = load_model("best_ecg_cnn_model.h5")

# Evaluate on test dataset
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"🚀 Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"🚀 Final Test Loss: {test_loss:.4f}")

# Save final scaler for normalizing future data
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")
