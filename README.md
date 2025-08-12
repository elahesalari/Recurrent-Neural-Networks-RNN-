# Recurrent Neural Networks (RNN) for ECG Signal Detection and Identification

Detection and Identification of Electrocardiogram Signals using Recurrent Neural Networks(RNNs)

---

## Electrocardiogram (ECG) Overview

An electrocardiogram records the electrical activity of the heart during each contraction. When an electrical wave passes through the heart, the interior of the heart cells becomes positive relative to the exterior, reflecting cellular depolarization. This electrical stimulation is critical for understanding heart function.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91370511/159136511-ce4524ab-a0e1-45a9-b02e-2ba13cb9c808.png" alt="ECG signal example" />
</p>

The **QRS complex** is the most important feature of the ECG signal used by cardiologists for diagnosing heart diseases. It is more distinct and easier to identify than other signal components such as the P and T waves. The QRS complex represents ventricular depolarization, a key electrical activity of the heart. Accurate detection and classification of the QRS complex are essential for diagnosing cardiac abnormalities and assessing heartbeat intensity and irregularities.

---

## Dataset: QT Database

The QT database includes 146 files, each containing:

- A primary sequence (`mV` signal values) representing the ECG waveform  
- A secondary sequence (`annotation`) with corresponding labels:
  - 0: Other signal points  
  - 1: P wave  
  - 2: QRS complex  
  - 3: T wave

Each file contains continuous data streams of these primary and secondary sequences. An example input-output sequence is shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/91370511/159136598-275f628a-517e-4453-a391-255365e108d5.png" alt="ECG sample input-output sequences" />
</p>

---

## Implementation Summary

1. Split each sequence file into **80% training** and **20% testing** portions for both primary and secondary sequences.  
2. Generate training samples using a sliding window approach with window size 11.  
3. Design, train, and evaluate an **Elman Neural Network** on the prepared data.  
4. Design, train, and evaluate a **NARX Neural Network** on the same data.  
5. Report training and testing accuracy for both models.  
6. Repeat the process using sliding window sizes of **5** and **21** to analyze the effect of window size on model performance.
