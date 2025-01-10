
# Autism Prediction using Machine Learning

This repository contains the `Autism_Prediction_using_Machine_Learning.ipynb` notebook, which implements a machine learning pipeline to predict Autism Spectrum Disorder (ASD) based on behavioral and demographic data. The project explores various models and techniques to ensure robust and accurate predictions.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook Structure](#notebook-structure)
- [Models Used](#models-used)
- [Results](#results)
- [Model Performance Summary](#model-performance-summary)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Autism Spectrum Disorder (ASD) is a developmental condition that affects communication, social interaction, and behavior. Early detection and diagnosis can improve outcomes significantly. This project utilizes a machine learning approach to analyze questionnaire and demographic data to predict ASD.

---

## Features

- Comprehensive data preprocessing pipeline
- Exploratory Data Analysis (EDA) and visualization
- Multiple machine learning models
- Performance metrics including accuracy, precision, recall, and F1-score
- Easy-to-follow notebook structure

---

## Dataset

The dataset contains behavioral screening data and demographic features. Some key columns include:

- Age
- Gender
- Ethnicity
- Family history of autism
- Responses to screening questions
- Target label (ASD or non-ASD)

### Note:
Ensure proper access and permissions for the dataset used. Place the dataset in the `data/` directory before running the notebook.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Autism_Prediction_using_Machine_Learning.git
   cd Autism_Prediction_using_Machine_Learning
   ```
2. Install Jupyter Notebook:
   ```bash
   pip install notebook
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Open the notebook:
   ```bash
   jupyter notebook Autism_Prediction_using_Machine_Learning.ipynb
   ```
2. Run each cell in sequence to:
   - Load and preprocess data
   - Perform exploratory data analysis
   - Train machine learning models
   - Evaluate model performance

3. View results and visualizations directly within the notebook.

---

## Notebook Structure

1. **Introduction**
   - Overview of the project
2. **Dataset Loading and Preprocessing**
   - Handling missing values, encoding categorical features, and scaling
3. **Exploratory Data Analysis**
   - Visualizations of key features and correlations
4. **Model Training and Evaluation**
   - Implementation of multiple models with evaluation metrics
5. **Results and Insights**
   - Analysis of performance and feature importance

---

## Models Used

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting (e.g., XGBoost, LightGBM)
- K-Nearest Neighbors
- Naive Bayes

---

## Results

### Model Results Overview:

- **Logistic Regression:**
  - Training Accuracy: 88%, Validation Accuracy: 87%
  - Training F1 Score: 0.70, Validation F1 Score: 0.70

- **Random Forest:**
  - Training Accuracy: 100%, Validation Accuracy: 84%
  - Training F1 Score: 1.00, Validation F1 Score: 0.63

- **XGBoost:**
  - Training Accuracy: 100%, Validation Accuracy: 84%
  - Training F1 Score: 1.00, Validation F1 Score: 0.62

- **Support Vector Machine:**
  - Training Accuracy: 90%, Validation Accuracy: 86%
  - Training F1 Score: 0.74, Validation F1 Score: 0.66

- **K-Nearest Neighbors:**
  - Training Accuracy: 90%, Validation Accuracy: 88%
  - Training F1 Score: 0.74, Validation F1 Score: 0.72

- **Naive Bayes:**
  - Training Accuracy: 83%, Validation Accuracy: 84%
  - Training F1 Score: 0.66, Validation F1 Score: 0.71

---


---
