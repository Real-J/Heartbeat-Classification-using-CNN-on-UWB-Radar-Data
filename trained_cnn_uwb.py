import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt
import joblib
import matplotlib.pyplot as plt

# ---------------- STEP 1: LOAD TRAINED MODEL ----------------
print("Loading trained CNN model...")
cnn_model = load_model("")

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")

# Define class labels
class_labels = {0: "Normal (N)", 1: "Supraventricular (S)", 2: "Ventricular (V)", 3: "Fusion (F)", 4: "Unknown (Q)"}

# ---------------- STEP 2: LOAD RAW UWB DATA ----------------
print("Loading raw UWB dataset...")
uwb_file = ""  # Update with your UWB dataset file path
df_uwb = pd.read_csv(uwb_file)

# Assume the second column contains the UWB signal
uwb_signal = df_uwb.iloc[:, 1].values  
fs = 1000  # Sampling frequency of UWB radar (adjust if needed)

# ---------------- STEP 3: FILTER UWB SIGNAL (1-3 Hz) ----------------
print("Filtering UWB signal for heartbeat frequencies (1-3 Hz)...")

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

filtered_uwb_signal = bandpass_filter(uwb_signal, 1, 3, fs, order=4)

# ---------------- STEP 4: SEGMENT UWB SIGNAL INTO ECG-SIZED WINDOWS ----------------
print("Segmenting UWB signal into ECG-sized windows...")

window_size = 187  # Same size as ECG segments used for training
uwb_segments = []
for i in range(0, len(filtered_uwb_signal) - window_size, window_size):
    segment = filtered_uwb_signal[i:i + window_size]
    uwb_segments.append(segment)

uwb_segments = np.array(uwb_segments)

# Normalize UWB segments using the same scaler as ECG training
uwb_segments = scaler.transform(uwb_segments)  
uwb_segments = uwb_segments.reshape(uwb_segments.shape[0], uwb_segments.shape[1], 1)

# ---------------- STEP 5: CLASSIFY UWB HEARTBEATS USING CNN ----------------
print("Classifying UWB segments using CNN...")

predictions = cnn_model.predict(uwb_segments)
predicted_classes = np.argmax(predictions, axis=1)

# ---------------- STEP 6: DISPLAY RESULTS ----------------
print("\nHeartbeat Classification Results:")
for i, pred in enumerate(predicted_classes[:10]):  # Display first 10 results
    print(f"Segment {i+1}: Classified as {class_labels[pred]}")

# Plot a few detected heartbeat waveforms
plt.figure(figsize=(12, 6))
for i in range(min(5, len(uwb_segments))):  # Plot first 5 detected heartbeats
    plt.plot(uwb_segments[i].flatten(), label=f"Heartbeat {i+1} ({class_labels[predicted_classes[i]]})")

plt.title("Extracted Heartbeats from UWB Radar")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

print(f"Total detected heartbeats classified: {len(predicted_classes)}")
