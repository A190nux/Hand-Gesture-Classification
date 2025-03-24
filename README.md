# Hand Gesture Recognition using MediaPipe Landmarks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A190nux/Hand-Gesture-Classification/blob/main/train_model.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest%20%7C%20SVM-orange)

A machine learning pipeline for classifying hand gestures from MediaPipe landmarks extracted from the HaGRID dataset.

## Project Overview

This project implements a robust hand gesture recognition system using:
- **MediaPipe** for hand landmark detection
- **HaGRID Dataset** containing 18 gesture classes
- Machine learning models (XGBoost, Random Forest, SVM) for classification
- Temporal smoothing for stable video predictions

**Key Features**:
- Data preprocessing pipeline for landmark normalization
- Comparative analysis of multiple ML models
- Real-time webcam prediction with smoothing
- Video recording capabilities

## Dataset Details

The [HaGRID dataset](https://github.com/hukenovs/hagrid) contains 18 gesture classes:
like, dislike, stop, peace, fist, four, call, mute, ok, peace_inverted, two_up, two_up_inverted, three, three2, four_inverted, rock, palm

Each sample contains:
- 21 hand landmarks (x, y, z coordinates)
- Preprocessed using wrist-centered normalization
- Mid-finger tip scaling for size invariance
