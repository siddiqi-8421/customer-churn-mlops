# %% [markdown]
# # Customer Churn — EDA & Statistical Foundations
# Open this in Jupyter or VS Code (each `# %%` is a runnable cell).

# %%
import sys
sys.path.append("../src")
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from data_prep import load_raw, clean

pd.set_option("display.max_columns", 50)

# %%
df = clean(load_raw("data/raw/telco_churn.csv"))
df.shape, df["Churn"].mean()

# %% [markdown]
# ## 1. Probability: base rate of churn
# Before any modeling, know your prior. A model that always predicts "no churn"
# already gets accuracy == 1 - churn_rate — this is *why* AUC/F1 matter more than accuracy here.

# %%
churn_rate = df["Churn"].mean()
print(f"Base churn rate: {churn_rate:.2%}")
print(f"A naive 'always predict no churn' model would score {1-churn_rate:.2%} accuracy")

# %% [markdown]
# ## 2. Distribution checks
# Is tenure normally distributed, or skewed? This affects whether we should
# use a t-test (assumes normality) or a non-parametric test (Mann-Whitney U).

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    df[col].hist(bins=30, ax=ax)
    ax.set_title(f"{col} (skew={stats.skew(df[col]):.2f})")
plt.tight_layout()

# %% [markdown]
# ## 3. Hypothesis testing: what actually drives churn?
# Chi-square test for categorical association; Welch's t-test for numeric group differences.
# Null hypothesis: the feature has no relationship with churn.

# %%
def chi_square_test(df, col, target="Churn"):
    table = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return chi2, p

for col in ["Contract", "InternetService", "PaymentMethod", "gender", "SeniorCitizen"]:
    chi2, p = chi_square_test(df, col)
    verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"{col:20s} chi2={chi2:8.2f}  p={p:.5f}  -> {verdict}")

# %%
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    churned = df[df.Churn == 1][col]
    retained = df[df.Churn == 0][col]
    t_stat, p = stats.ttest_ind(churned, retained, equal_var=False)
    verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"{col:15s} t={t_stat:7.2f}  p={p:.5f}  -> {verdict}  "
          f"(churned mean={churned.mean():.1f}, retained mean={retained.mean():.1f})")

# %% [markdown]
# ## 4. Correlation structure (numeric features)

# %%
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr(numeric_only=True)
corr["Churn"].sort_values(ascending=False)

# %% [markdown]
# ## Takeaways to carry into feature engineering (src/features.py)
# - Month-to-month contracts and short tenure are strongly associated with churn (p < 0.05)
# - TotalCharges is right-skewed — consider log-transform or the ratio features in features.py
# - Fiber-optic internet customers churn more than DSL — worth a categorical interaction feature