
# Heart Disease Prediction

## Overview
This project aims to predict the presence of heart disease in patients using a dataset of various health metrics. By leveraging machine learning algorithms and extensive data preprocessing, the project focuses on ensuring high recall for identifying individuals with heart disease, a crucial aspect in medical diagnostics.

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Features](#features)
- [Preprocessing Steps](#preprocessing-steps)
- [Modeling](#modeling)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Objectives
1. **Explore the Dataset**: Analyze patterns, distributions, and relationships in the data.
2. **Conduct Exploratory Data Analysis (EDA)**: Perform univariate and bivariate analyses to understand feature-target relationships.
3. **Preprocess Data**:
   - Remove irrelevant features
   - Handle missing values
   - Treat outliers
   - Encode categorical variables
   - Transform skewed features
4. **Build Models**:
   - Establish pipelines for scaling and preprocessing
   - Train and tune machine learning models such as KNN, SVM, Decision Trees, and Random Forest
   - Prioritize high recall for the positive class
5. **Evaluate Models**: Use precision, recall, F1-score, and other metrics to assess performance.

## Dataset
The dataset contains 303 entries with 14 attributes. Each entry represents a patient and includes details like age, cholesterol levels, blood pressure, and heart disease status. The dataset is sourced from [Kaggle](https://www.kaggle.com/code/farzadnekouei/heart-disease-prediction).

## Features
- `age`: Age of the patient
- `sex`: Gender (0 = male, 1 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar (>120 mg/dl, 1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of the ST segment (0-2)
- `ca`: Number of major vessels (0-4)
- `thal`: Thalium stress test result (0-3)
- `target`: Heart disease status (1 = disease, 0 = no disease)

## Preprocessing Steps
1. Handle missing values (none detected).
2. Identify and treat outliers using the IQR method.
3. Encode categorical features (e.g., one-hot encoding for `cp`, `restecg`, and `thal`).
4. Scale features for distance-based models.
5. Apply transformations (e.g., Box-Cox) to normalize skewed features.

## Modeling
The following models were trained and tuned:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Trees (DT)
- Random Forest (RF)

## Evaluation Metrics
- **Precision**
- **Recall (focus on positive class)**: Critical for identifying all potential heart patients.
- **F1-Score**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/401DHARshini/Heart-Disease-Predictions.git
   cd heart-disease-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the preprocessing and EDA script:
   ```bash
   python preprocess.py
   ```
2. Train and evaluate models:
   ```bash
   python train_models.py
   ```

## Results
The SVM model achieved the highest recall for the positive class (97%), ensuring almost all heart disease cases were identified while maintaining balanced precision.

## Future Work
- Expand the dataset for better generalization.
- Incorporate deep learning models for improved accuracy.
- Develop a real-time prediction system.

## Contributors
- Devadarshini D - ((https://www.linkedin.com/in/devadarshini-duraisamy-b99409225/)

## License
This project is licensed under the [MIT License](LICENSE).

