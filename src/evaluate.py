"""
Evaluate model with statistics, not just one accuracy number.
Model ko statistics se check karo, sirf ek accuracy number se nahi.
"""
import joblib
import pandas as pd
from scipy import stats   # statistical tests | statistics ke test
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve, classification_report
)


def chi_square_churn_driver(df: pd.DataFrame, column: str, target: str = "Churn"):
    # test if category is linked to churn | check karo category churn se juda hai ya nahi
    contingency = pd.crosstab(df[column], df[target])  # count table | count table banao
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    return {"feature": column, "chi2": round(chi2, 2), "p_value": round(p_value, 5),
            "significant_at_0.05": p_value < 0.05}  # true if real effect | agar asli effect hai to true


def ttest_numeric_driver(df: pd.DataFrame, column: str, target: str = "Churn"):
    # test if number differs between churn groups | check karo number dono group me alag hai ya nahi
    churned = df[df[target] == 1][column]   # churned customers | jo customer chhod gaye
    retained = df[df[target] == 0][column]  # retained customers | jo customer ruke hain
    t_stat, p_value = stats.ttest_ind(churned, retained, equal_var=False)
    return {"feature": column, "t_stat": round(t_stat, 2), "p_value": round(p_value, 5),
            "churned_mean": round(churned.mean(), 2), "retained_mean": round(retained.mean(), 2)}


def evaluate_model(model_path: str = "models/churn_model.pkl", test_path: str = "data/processed/test.csv"):
    # pipeline already has feature engineering inside | pipeline ke andar feature engineering already hai
    model = joblib.load(model_path)       # load saved model | saved model load karo
    test_df = pd.read_csv(test_path)      # load raw test data | raw test data load karo

    X_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]
    proba = model.predict_proba(X_test)[:, 1]  # risk score 0 to 1 | risk score 0 se 1
    preds = model.predict(X_test)               # final 0/1 prediction | final 0/1 prediction

    print("=== Classification report ===")
    print(classification_report(y_test, preds))  # precision, recall, f1 | precision, recall, f1
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))  # overall ranking score | overall ranking score

    cm = confusion_matrix(y_test, preds)  # 4-box error table | 4-box error table
    print("Confusion matrix:")
    print(f"                 Predicted: No   Predicted: Yes")
    print(f"Actual: No       {cm[0][0]:<15} {cm[0][1]}")
    print(f"Actual: Yes      {cm[1][0]:<15} {cm[1][1]}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)  # for threshold tuning | threshold tune karne ke liye
    return precisions, recalls, thresholds


if __name__ == "__main__":
    raw = pd.read_csv("data/processed/train.csv")  # use train set for stats | stats ke liye train set

    print("=== Statistical drivers of churn (train set) ===\n")
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        print(chi_square_churn_driver(raw, col))  # test categories | categories test karo
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        print(ttest_numeric_driver(raw, col))      # test numbers | numbers test karo

    print("\n=== Model evaluation on held-out test set ===\n")
    evaluate_model()  # run final evaluation | final evaluation chalao