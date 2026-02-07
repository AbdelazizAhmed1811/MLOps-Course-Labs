# Bank Customer Churn Prediction with MLflow

A machine learning project for predicting bank customer churn using various classification models, with experiment tracking via **MLflow**.

## 📋 Overview

This project implements customer churn prediction using three different models:
- **Logistic Regression** - Baseline model
- **Random Forest** - Ensemble tree-based model
- **Gradient Boosting** - Boosted decision trees

All experiments are tracked using MLflow, including:
- Model parameters
- Performance metrics (accuracy, precision, recall, F1-score)
- Model artifacts
- Confusion matrices

## 🏗️ Project Structure

```
MLOps-Course-Labs/
├── dataset/
│   └── Churn_Modelling.csv       # Bank customer dataset
├── src/
│   ├── train.py                  # Logistic Regression training
│   ├── train_random_forest.py    # Random Forest training
│   └── train_gradient_boosting.py # Gradient Boosting training
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Start MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Run Training Scripts

**Logistic Regression:**
```bash
python src/train.py
```

**Random Forest:**
```bash
python src/train_random_forest.py
```

**Gradient Boosting:**
```bash
python src/train_gradient_boosting.py
```

## 📊 Model Comparison

| Model | Key Parameters |
|-------|----------------|
| Logistic Regression | `max_iter=1000` |
| Random Forest | `n_estimators=100`, `max_depth=10` |
| Gradient Boosting | `n_estimators=150`, `learning_rate=0.1`, `max_depth=5` |

## 🔍 Viewing Results

After running the training scripts, open the MLflow UI at:

```
http://localhost:5000
```

Navigate to the experiment `churn_prediction_experiment-01` to compare runs and metrics.

## 📁 Logged Artifacts

Each run logs:
- Trained model (sklearn format)
- Column transformer (`col_transf.pkl`)
- Confusion matrix image

## 📈 Metrics Tracked

- **Accuracy** - Overall correct predictions
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1 Score** - Harmonic mean of precision and recall

## 🛠️ Dataset Features

| Feature | Description |
|---------|-------------|
| CreditScore | Customer credit score |
| Geography | Customer location |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years as customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Active member status (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | Target - Churned (1) or Not (0) |

## 📝 License

This project is for educational purposes as part of the MLOps course.
