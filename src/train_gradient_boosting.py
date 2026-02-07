from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score
from train import preprocess

import mlflow

def train(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5):
    """Train a Gradient Boosting model."""
    gb_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Infer signature
    predictions = gb_model.predict(X_train)
    signature = mlflow.models.infer_signature(X_train, predictions)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=gb_model,
        artifact_path="gradient_boosting_model",
        signature=signature
    )

    # Log the data
    dataset = mlflow.data.from_pandas(
        X_train,
        source="data.csv",
        name="training_features"
    )
    mlflow.log_input(dataset, context="training")

    return gb_model


def main():
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('churn_prediction_experiment-gradient_boosting')

    n_estimators = 150
    learning_rate = 0.1
    max_depth = 5

    with mlflow.start_run(run_name="GradientBoosting_Run") as run:
        df = pd.read_csv("/home/zizo/iti/MLflow/MLOps-Course-Labs/dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "GradientBoosting")

        model = train(X_train, y_train, n_estimators, learning_rate, max_depth)

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
        mlflow.set_tag("model_type", "GradientBoosting")

        # Confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix_gb.png")
        mlflow.log_artifact("confusion_matrix_gb.png")

        print(f"Run completed! Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        plt.close()


if __name__ == "__main__":
    main()
