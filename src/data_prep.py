"""
Data loading and cleaning for Telco Customer Churn dataset.
Telco Customer Churn dataset ko load aur clean karne ka code.
"""
import pandas as pd        # data handling library | data handle karne ki library
import numpy as np         # numeric operations | number ka kaam


def load_raw(path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)                           # read CSV file | CSV file padho
    df.columns = [c.strip() for c in df.columns]     # remove extra spaces | extra space hatao
    return df                                        # return cleaned dataframe | saaf dataframe wapas do


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # work on a copy, don't touch original | copy pe kaam karo, original mat chhuo

    # TotalCharges has blanks for new customers | naye customer ke liye TotalCharges khaali hai
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # convert to number | number banao
    df["TotalCharges"] = df["TotalCharges"].fillna(0)  # fill blanks with 0 | khaali jagah 0 se bharo

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])  # ID is not useful for model | ID model ke kaam ki nahi

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  # convert Yes/No to 1/0 | Yes/No ko 1/0 banao

    # some columns say "No internet service" instead of "No" | kuch column me "No internet service" likha hai
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    for col in service_cols:
        if col in df.columns:
            # make it simple: just "No" | simple karo: sirf "No" likho
            df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    return df  # return cleaned data | saaf data wapas do


def train_test_split_save(df: pd.DataFrame, out_dir: str = "data/processed", test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import train_test_split  # split function | split karne wala function

    # split data, keep churn ratio same in both parts | data split karo, churn ratio dono me same rakho
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["Churn"]
    )
    train_df.to_csv(f"{out_dir}/train.csv", index=False)  # save train data | train data save karo
    test_df.to_csv(f"{out_dir}/test.csv", index=False)    # save test data | test data save karo

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")  # show sizes | size dikhao
    print(f"Churn rate — train: {train_df['Churn'].mean():.3f}, test: {test_df['Churn'].mean():.3f}")  # show churn % | churn % dikhao
    return train_df, test_df


if __name__ == "__main__":
    raw = load_raw()              # step 1: load | step 1: load karo
    cleaned = clean(raw)          # step 2: clean | step 2: saaf karo
    train_test_split_save(cleaned)  # step 3: split and save | step 3: split karke save karo