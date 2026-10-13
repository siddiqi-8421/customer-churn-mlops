"""
Feature engineering as a scikit-learn transformer.
Scikit-learn transformer ke roop me naye features banana.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin  # base classes for pipeline | pipeline ke liye base class

# fixed list of numeric columns | number wale column ki fix list
NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_charge_per_month", "charge_ratio", "num_addon_services", "is_month_to_month",
]
# fixed list of category columns | category wale column ki fix list
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_bucket",
]
CATEGORICAL_FEATURES.append("SeniorCitizen")  # add senior citizen too | senior citizen bhi jodo


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """No learning happens here, only fixed rules.
    Yahan kuch seekha nahi jaata, sirf fix rules chalte hain."""

    def fit(self, X, y=None):
        return self  # nothing to learn, just return self | kuch seekhna nahi, bas self wapas do

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()  # copy input data | input data ki copy banao

        # group tenure into buckets | tenure ko groups me baanto
        df["tenure_bucket"] = pd.cut(
            df["tenure"], bins=[-1, 6, 12, 24, 48, np.inf],
            labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4yr+"]
        ).astype(str)

        # average monthly spend so far | ab tak ka average monthly kharch
        df["avg_charge_per_month"] = np.where(
            df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
        )

        # is current bill higher than usual? | kya abhi ka bill normal se zyada hai?
        df["charge_ratio"] = df["MonthlyCharges"] / df["avg_charge_per_month"].replace(0, np.nan)
        df["charge_ratio"] = df["charge_ratio"].fillna(1.0)  # fill missing with 1 | khaali jagah 1 se bharo

        # count how many extra services customer has | customer ke paas kitni extra service hai
        addon_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        existing = [c for c in addon_cols if c in df.columns]
        df["num_addon_services"] = (df[existing] == "Yes").sum(axis=1)  # count "Yes" values | "Yes" count karo

        # flag risky contract type | risky contract ko flag karo
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

        return df  # return data with new columns | naye column ke saath data wapas do

    def get_feature_names_out(self, input_features=None):
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES  # list of all output columns | sab output column ki list


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return ChurnFeatureEngineer().fit_transform(df)  # shortcut function | shortcut function


def get_feature_lists(df: pd.DataFrame = None):
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES  # return both lists | dono list wapas do


if __name__ == "__main__":
    df = pd.read_csv("data/processed/train.csv")       # load train data | train data load karo
    engineered = ChurnFeatureEngineer().fit_transform(df)  # run transformer | transformer chalao
    print(engineered[["tenure", "tenure_bucket", "avg_charge_per_month",
                       "charge_ratio", "num_addon_services", "is_month_to_month"]].head())
    print("\nShape:", engineered.shape)  # show new shape | naya shape dikhao