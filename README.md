# Hybrid-CNN-InSAR-Subsidence-Estimation
Enhanced Land Subsidence Interpolation through a Hybrid Deep Convolutional Neural Network and InSAR Time Series
This repository contains the official implementation of the hybrid deep‑learning method introduced in:

“Enhanced land subsidence interpolation through a hybrid deep convolutional neural network and InSAR time series”  
Zahra Azarm, Hamid Mehrabi, Saeed Nadi  
Geoscientific Model Development, 2025  
https://doi.org/10.5194/gmd-18-6903-2025 (doi.org in Bing)

“This study introduces a hybrid approach that combines deep convolutional neural networks (CNNs) with persistent scatterer interferometric synthetic aperture radar (PSInSAR) to estimate land subsidence in areas where PSInSAR data are unreliable or sparse.”
📌 Overview
Land subsidence poses a major global threat to infrastructure and the environment. Traditional interpolation methods (Kriging, IDW, RBF) struggle to produce continuous subsidence surfaces due to the sparse and uneven distribution of PSInSAR points.

This project implements a hybrid CNN + PSInSAR model that:

Learns subsidence patterns from PSInSAR measurements

Incorporates nine driving forces (NDVI, slope, SPI, TWI, groundwater depth, land use, etc.)

Predicts continuous subsidence surfaces even in areas with no PS points

Achieves dramatically lower error compared to classical interpolation

“The model achieved RMSEs of 3.99, 8.47, and 9 mm… while kriging, IDW, and RBF yielded RMSE values of 61.60, 66.21, and 61.76 mm.”

🚀 Key Contributions
A 31‑layer CNN architecture designed for geospatial regression

Input patches of 30×30×9 representing spatial neighborhoods + driving forces

Hyperparameter tuning for optimal performance

Full workflow: PSInSAR → Driving Forces → CNN Training → Prediction

85% improvement over traditional interpolation methods

🧠 CNN Architecture Summary
Your CNN includes:

1×1 and 3×3 convolutional layers

Batch normalization

Dropout (0.1)

Max‑pooling

Fully connected layers: 1024 → 512 → 256 → 1

ReLU activations (hidden layers)

Linear activation (output)

“The CNN has 31 layers… including three 1×1 convolutional layers, three 3×3 convolutional layers… and fully connected layers with 1024, 512, and 256 ReLU neurons.”

What the resulting map will look like:

Original raster
┌──────────────────────────────┐
│ N N N N N N N N N N N N N   │
│ N ┌──────────────────────┐ N │
│ N │                      │ N │
│ N │   CNN predictions    │ N │
│ N │                      │ N │
│ N └──────────────────────┘ N │
│ N N N N N N N N N N N N N   │
└──────────────────────────────┘

N = NoData

