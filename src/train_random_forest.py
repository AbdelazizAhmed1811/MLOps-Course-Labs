import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import mlflow
import joblib


from train import preprocess

def train(X_train, y_train, n_estimators=100, max_depth=10):
    """Train a Random Forest model."""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Infer signature
    predictions = rf_model.predict(X_train)
    signature = mlflow.models.infer_signature(X_train, predictions)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=rf_model,
        artifact_path="random_forest_model",
        signature=signature
    )

    # Log the data
    dataset = mlflow.data.from_pandas(
        X_train,
        source="data.csv",
        name="training_features"
    )
    mlflow.log_input(dataset, context="training")

    return rf_model


def main():
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('churn_prediction_experiment-random_forest')

    # Random Forest hyperparameters
    n_estimators = 100
    max_depth = 10

    with mlflow.start_run(run_name="RandomForest_Run") as run:
        df = pd.read_csv("/home/zizo/iti/MLflow/MLOps-Course-Labs/dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForest")

        model = train(X_train, y_train, n_estimators, max_depth)

        y_pred = model.predict(X_test)

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log tag
        mlflow.set_tag("model_type", "RandomForest")

        # Confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix_rf.png")
        mlflow.log_artifact("confusion_matrix_rf.png")

        print(f"Run completed! Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        plt.close()


if __name__ == "__main__":
    main()
