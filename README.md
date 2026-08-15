# Machine-Learning-Projects
A collection of Machine Learning projects including Titanic, Mohamed Salah, and Liverpool datasets.
# Phase 1: Titanic Survival Prediction 🚢

A Machine Learning classification project to predict passenger survival on the Titanic dataset using Python.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing & Cleaning](#data-preprocessing--cleaning)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics](#evaluation-metrics)

---

## 🔍 Project Overview
The goal of this project is to build and evaluate classification models to determine whether a passenger survived the Titanic disaster based on demographic and trip-related features (such as age, sex, passenger class, and fare).

---

## 🧹 Data Preprocessing & Cleaning
- **Missing Values Imputation:** 
  - Filled missing values in `Age` and `Fare` using the **median**.
  - Filled missing values in `Embarked` (port of embarkation) using the **mode**.
- **Feature Dropping:** Dropped columns with excessive missing data or irrelevant info (`Cabin`, `Name`, `Ticket`).
- **Categorical Encoding:** Applied **One-Hot Encoding** (`pd.get_dummies`) on categorical features (`Sex`, `Embarked`, `Pclass`).
- **Data Splitting:** Split the dataset into Train, Validation, and Test sets (70% train, 15% validation, 15% test) with stratification.
- **Feature Scaling:** Standardized numerical features using `StandardScaler` for the Logistic Regression model.

---

## 🤖 Machine Learning Models
Two classification models were trained and evaluated:
1. **Logistic Regression (Baseline Model):** Trained on scaled numerical and encoded categorical features.
2. **Random Forest Classifier (Advanced Model):** Trained with an ensemble of 200 estimators (`n_estimators=200`).

---

## 📈 Evaluation & Results
Both models were evaluated on the test set using multiple metrics:
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix & ROC Curves** (Visualized using Seaborn and Matplotlib)
- # Phase 2: Liverpool Goals Prediction (Regression) ⚽

A Machine Learning regression project to predict match goals for Liverpool FC based on performance metrics and advanced match statistics.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Data Parsing & Preprocessing](#data-preprocessing--preprocessing)
- [Feature Engineering & Outlier Handling](#feature-engineering--outlier-handling)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation & Performance Metrics](#evaluation--performance-metrics)

---

## 🔍 Project Overview
The objective of this project is to predict the number of goals scored (`Gls`) in matches using statistical metrics such as shots (`Sh`), shots on target (`SoT`), expected goals (`xG`), non-penalty expected goals (`npxG`), and shot distance (`Dist`).

---

## 🛠️ Data Parsing & Preprocessing
- **Raw Text Processing:** Extracted structured match records from raw text files using Regular Expressions (`re`).
- **Data Cleaning:** Handled missing columns, converted data types into numeric values, and removed rows with missing target or feature values.
- **Data Splitting:** Split the dataset into Train, Validation, and Test sets (60% train, 20% validation, 20% test).

---

## ⚙️ Feature Engineering & Outlier Handling
- **New Feature Creation:** Engineered a new metric `SoT_per_Sh` (Ratio of Shots on Target to Total Shots) to capture shooting accuracy.
- **Outlier Treatment:** Applied Interquartile Range (IQR) clipping (`iqr_clip`) to handle anomalous values across features and target variables safely.

---

## 🤖 Machine Learning Models & Tuning
Two regression models were developed and compared:
1. **Linear Regression (Baseline):** Trained on standardized features using `StandardScaler`.
2. **Random Forest Regressor (Advanced & Tuned):** Optimized using **GridSearchCV** with 5-fold cross-validation to find the best hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).

---

## 📈 Evaluation & Results
Models were evaluated on both validation and test sets using standard regression metrics:
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**
- **Visualizations:** Generated Predicted vs. Actual scatter plots and Residual Plots for both models to analyze error distribution.
- # Phase 3: Mohamed Salah Goals Time Series Forecasting ⚽📈

A deep learning project to forecast Mohamed Salah's match goals over time using sequential models (LSTM and BiLSTM) with TensorFlow and Keras.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing & Time-Based Split](#data-preprocessing--time-based-split)
- [Sequence Generation & Scaling](#sequence-generation--scaling)
- [Deep Learning Models (LSTM & BiLSTM)](#deep-learning-models-lstm--bilstm)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation & Results](#evaluation--results)

---

## 🔍 Project Overview
The objective of this phase is to treat match statistics as a time series and forecast goals scored across matches. We compare a Naive baseline model against advanced deep recurrent architectures (**LSTM** and **Bidirectional LSTM**).

---

## 🛠️ Data Preprocessing & Time-Based Split
- **Date Sorting:** Converted date columns into datetime objects and chronologically sorted the matches.
- **Data Cleaning:** Removed non-numeric and unplayed match rows to clean the target series (`Gls`).
- **Temporal Splitting:** Split the sequence chronologically into Training (70%), Validation (15%), and Test (15%) sets to prevent data leakage.

---

## ⚙️ Sequence Generation & Scaling
- **MinMax Scaling:** Scaled the target values using `MinMaxScaler` fitted strictly on the training data.
- **Sliding Windows:** Created sliding window sequences with different window sizes (`w = 3, 5, 7`) to feed past match data into the networks.

---

## 🤖 Deep Learning Models
1. **LSTM Model:** Built with a 64-unit LSTM layer followed by a Dense output layer, optimized using Adam optimizer and MSE loss.
2. **Bidirectional LSTM (BiLSTM):** Enhanced with a Bidirectional wrapper and `EarlyStopping` callbacks to capture patterns from past and future context within the window, preventing overfitting.

---

## 🔬 Hyperparameter Tuning
- Automated grid-style search over window sizes (`3, 5, 7`), hidden units (`32, 64`), and learning rates (`1e-3, 5e-4`) evaluated on validation RMSE.

---

## 📈 Evaluation & Results
- **Metrics:** Evaluated models on the test set using RMSE, MAE, and R² Score, compared against an aligned Naive baseline.
- **Visualizations:** Plotted Actual vs. Predicted trajectories, Absolute Error curves, and Residual Plots.
- **Artifacts Saved:** Trained models (`.keras`), scalers (`.joblib`), and JSON performance summaries were successfully exported.
