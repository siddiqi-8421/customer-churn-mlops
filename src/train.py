"""
Train models, tune hyperparameters, track with MLflow.
Model train karo, settings tune karo, MLflow se track karo.
"""
import joblib               # save/load model | model save/load karna
import mlflow                # experiment tracking | experiment track karna
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer           # combine preprocessing steps | preprocessing jodna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import ChurnFeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 8, 12, None]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3, 4]},
    ),
}


def build_pipeline(estimator):
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("feature_eng", ChurnFeatureEngineer()),
        ("preprocess", preprocess),
        ("clf", estimator),
    ])


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train, y_train = train_df.drop(columns=["Churn"]), train_df["Churn"]
    X_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]

    mlflow.set_experiment("customer-churn-prediction")
    best_auc, best_model, best_name = -1, None, None

    for name, (estimator, param_grid) in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipe = build_pipeline(estimator)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            search = RandomizedSearchCV(
                pipe, param_grid, n_iter=6, scoring="roc_auc",
                cv=cv, random_state=42, n_jobs=-1,
            )
            search.fit(X_train, y_train)

            preds = search.predict(X_test)
            proba = search.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            f1 = f1_score(y_test, preds)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_best_score_auc", search.best_score_)
            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("test_f1", f1)

            # trust our custom class + sklearn's internal tree type | apni class + sklearn ka internal tree type trust karo
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
                skops_trusted_types=[
                    "features.ChurnFeatureEngineer",
                    "sklearn.tree._tree.Tree",
                ],
            )

            print(f"\n=== {name} ===")
            print(f"CV best AUC: {search.best_score_:.4f} | Test AUC: {auc:.4f} | Test F1: {f1:.4f}")
            print(classification_report(y_test, preds))

            if auc > best_auc:
                best_auc, best_model, best_name = auc, search.best_estimator_, name

    joblib.dump(best_model, "models/churn_model.pkl")
    print(f"\nBest model: {best_name} (test AUC = {best_auc:.4f}) saved to models/churn_model.pkl")


if __name__ == "__main__":
    main()