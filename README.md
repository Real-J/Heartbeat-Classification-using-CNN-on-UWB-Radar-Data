# Heartbeat Classification using CNN on UWB Radar Data

## Overview
This project uses a pre-trained Convolutional Neural Network (CNN) to classify heartbeats detected from Ultra-Wideband (UWB) radar signals. The UWB data is processed to extract heartbeat-related signals, which are then normalized and fed into the CNN for classification.

## Features
- Loads a trained CNN model for heartbeat classification.
- Processes raw UWB signals by applying a bandpass filter (1-3 Hz).
- Segments UWB data into ECG-sized windows (187 samples per segment).
- Normalizes the data using a pre-trained scaler.
- Classifies each heartbeat segment into five categories:
  - Normal (N)
  - Supraventricular (S)
  - Ventricular (V)
  - Fusion (F)
  - Unknown (Q)
- Displays classification results and visualizes detected heartbeats.

## Definitions
### 1. **CNN (Convolutional Neural Network)**
A type of deep learning model that is particularly effective for analyzing time-series and spatial data, such as ECG waveforms. It extracts important patterns and features from input signals.

### 2. **UWB (Ultra-Wideband) Radar**
A wireless communication and sensing technology that uses short pulses over a wide frequency range to detect objects, movements, and vital signs such as heartbeats.

### 3. **ECG (Electrocardiogram)**
A recording of the electrical activity of the heart, commonly used for diagnosing cardiac conditions. The CNN model in this project was trained on ECG data to recognize heartbeat patterns.

### 4. **Bandpass Filtering**
A signal processing technique that allows signals within a specific frequency range (1-3 Hz in this case) to pass through while removing noise and irrelevant frequencies.

### 5. **Normalization**
A data preprocessing step where values are scaled to a common range (e.g., between -1 and 1) so that the model processes them effectively and consistently.

### 6. **Segmentation**
Dividing a continuous signal into fixed-size windows (187 samples in this case) to extract individual heartbeats for classification.

### 7. **Scaler**
A pre-trained model that was used to standardize the ECG data during training. The same scaler is used to normalize new UWB signal inputs before feeding them into the CNN.

### 8. **Classification**
The process of assigning labels to input data based on learned patterns. The CNN predicts the heartbeat type (N, S, V, F, or Q) for each segment.

## Model Performance and Training Summary
### **1. Model Performance During Training**
- **Final Training Accuracy:** **99.17%** (Epoch 20)
- **Final Training Loss:** **0.0243**
- **Validation Accuracy:** Fluctuated slightly but remained around **99.05%-99.17%**.
- **Best Model Saved at Epoch 20:** Model improved in the last epoch, leading to the best accuracy being saved at **99.17%**.

📌 **What This Means:**  
- A **99.17% accuracy** suggests that the model **rarely makes mistakes** on training data.
- The **low loss (0.0243)** indicates high confidence in predictions.
- Stable performance across epochs, avoiding overfitting.

### **2. Model Performance on Test Data**
- **Final Test Accuracy:** **98.47%**
- **Final Test Loss:** **0.0975**

📌 **What This Means:**  
- A **98.47% accuracy on unseen test data** shows strong generalization.
- The **slightly higher loss (0.0975) compared to training loss (0.0243)** is expected in real-world applications.
- The small accuracy drop from **99.17% (training) → 98.47% (test)** suggests **minimal overfitting**.

### **3. Key Observations**
✅ **Consistently High Accuracy:** The model maintained **99% training accuracy** and **98.47% test accuracy**, showing strong learning capabilities.  
✅ **Minimal Overfitting:** The small gap between training and test accuracy suggests the model didn't just memorize the training data but **learned meaningful patterns**.  
✅ **Efficient Training Progression:** Improved over **20 epochs**, showing good learning behavior.  
✅ **Best Model Saved Automatically:** The best-performing model was saved (`best_ecg_cnn_model.h5`) based on validation accuracy.

## Installation
### Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas tensorflow scipy joblib matplotlib
```


## Explanation of the Pipeline
1. **Load the trained CNN model**: Uses a previously trained model on ECG data.
2. **Read the UWB signal**: Loads the raw UWB radar signal from a CSV file.
3. **Filter the signal**: Applies a bandpass filter (1-3 Hz) to extract heartbeat components.
4. **Segment into ECG-sized windows**: Splits the signal into segments of 187 samples.
5. **Normalize the data**: Uses a pre-trained scaler to match the CNN’s expected input format.
6. **Classify the heartbeats**: Feeds segments into the CNN for classification.
7. **Display results**: Prints predictions and visualizes heartbeat waveforms.

## Example Output
```
Loading trained CNN model...
Loading raw UWB dataset...
Filtering UWB signal for heartbeat frequencies (1-3 Hz)...
Segmenting UWB signal into ECG-sized windows...
Classifying UWB segments using CNN...

Heartbeat Classification Results:
Segment 1: Classified as Normal (N)
Segment 2: Classified as Ventricular (V)
...
Total detected heartbeats classified: 5
```

## Visualization
The script plots detected heartbeats with their classifications:
![Heartbeat Visualization](example_plot.png)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Credits

The UWB dataset was provided by Moro, G.; Di Luca, F.; Dardari, D.; Frisoni, G. Human Being Detection from UWB NLOS Signals: Accuracy and Generality of Advanced Machine Learning Models. Sensors 2022, 22, 1656. https://doi.org/10.3390/s22041656.

The CNN was trained with the pubilcally available ECG dataset (https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

