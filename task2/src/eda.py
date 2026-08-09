import os
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ==========================================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "titanic.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================================
# 1. LOAD DATA & INITIAL INSPECTION
# ==========================================================

print("=" * 60)
print("1. DATA LOADING & INITIAL INSPECTION")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nData Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================================
# 2. SUMMARY STATISTICS
# ==========================================================

print("\n" + "=" * 60)
print("2. SUMMARY STATISTICS")
print("=" * 60)

print(df.describe(include="all"))


# ==========================================================
# 3. HANDLE MISSING VALUES
# ==========================================================

print("\n" + "=" * 60)
print("3. MISSING VALUE HANDLING")
print("=" * 60)

# Make a copy for EDA
eda_df = df.copy()

# Age: fill missing values with median
if "Age" in eda_df.columns:
    eda_df["Age"] = eda_df["Age"].fillna(
        eda_df["Age"].median()
    )

# Embarked: fill missing values with mode
if "Embarked" in eda_df.columns:
    eda_df["Embarked"] = eda_df["Embarked"].fillna(
        eda_df["Embarked"].mode()[0]
    )

print("\nMissing values after imputation:")
print(eda_df.isnull().sum())


# ==========================================================
# 4. DISTRIBUTION ANALYSIS
# ==========================================================

print("\n" + "=" * 60)
print("4. DISTRIBUTION ANALYSIS")
print("=" * 60)


# ---------- Age Distribution ----------

plt.figure(figsize=(8, 5))

sns.histplot(
    eda_df["Age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_age_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------- Fare by Class ----------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Pclass",
    y="Fare",
    data=eda_df
)

plt.title("Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_fare_by_class.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------- Survival by Gender ----------

plt.figure(figsize=(8, 5))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=eda_df
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_survival_by_gender.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------- Survival by Passenger Class ----------

plt.figure(figsize=(8, 5))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=eda_df
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_survival_by_class.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 5. CORRELATION & RELATIONSHIPS
# ==========================================================

print("\n" + "=" * 60)
print("5. CORRELATION & RELATIONSHIPS")
print("=" * 60)

numeric_df = eda_df.select_dtypes(
    include=np.number
)

# ---------- Correlation Heatmap ----------

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_correlation_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------- Cross Tabulation ----------

class_survival = pd.crosstab(
    eda_df["Pclass"],
    eda_df["Survived"],
    normalize="index"
) * 100

print("\nSurvival Percentage by Passenger Class:")
print(class_survival.round(2))


gender_survival = pd.crosstab(
    eda_df["Sex"],
    eda_df["Survived"],
    normalize="index"
) * 100

print("\nSurvival Percentage by Gender:")
print(gender_survival.round(2))


# ---------- Pearson Correlation ----------

pearson_corr = eda_df["Fare"].corr(
    eda_df["Survived"],
    method="pearson"
)

print(
    f"\nPearson Correlation "
    f"(Fare vs Survival): {pearson_corr:.4f}"
)


# ---------- Spearman Correlation ----------

spearman_corr = eda_df["Fare"].corr(
    eda_df["Survived"],
    method="spearman"
)

print(
    f"Spearman Correlation "
    f"(Fare vs Survival): {spearman_corr:.4f}"
)


# ==========================================================
# 6. OUTLIER DETECTION
# ==========================================================

print("\n" + "=" * 60)
print("6. OUTLIER DETECTION")
print("=" * 60)


# ---------- Fare Boxplot ----------

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=eda_df["Fare"]
)

plt.title("Fare Outliers")
plt.xlabel("Fare")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_fare_outliers.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------- IQR Method ----------

Q1 = eda_df["Fare"].quantile(0.25)
Q3 = eda_df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outliers = eda_df[
    (eda_df["Fare"] < lower_bound)
    |
    (eda_df["Fare"] > upper_bound)
]

print("\nIQR Method")
print("Lower Bound:", round(lower_bound, 2))
print("Upper Bound:", round(upper_bound, 2))
print("Number of IQR Outliers:", len(iqr_outliers))


# ---------- Z-Score Method ----------

z_scores = np.abs(
    stats.zscore(eda_df["Fare"])
)

z_outliers = eda_df[
    z_scores > 3
]

print("\nZ-Score Method")
print(
    "Number of Z-Score Outliers:",
    len(z_outliers)
)

print(
    "\nDecision: Keep legitimate fare outliers "
    "for EDA because extreme fares may represent "
    "legitimate first-class tickets."
)


# ==========================================================
# 7. ADVANCED VISUALIZATION
# ==========================================================

print("\n" + "=" * 60)
print("7. ADVANCED VISUALIZATION")
print("=" * 60)


# ---------- FacetGrid ----------

g = sns.FacetGrid(
    eda_df,
    col="Survived",
    row="Pclass",
    height=3
)

g.map_dataframe(
    sns.histplot,
    x="Age",
    bins=20
)

g.set_axis_labels(
    "Age",
    "Number of Passengers"
)

g.fig.suptitle(
    "Age Distribution by Survival and Passenger Class",
    y=1.02
)

g.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_facet_age_analysis.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ---------- Pairplot ----------

pair_columns = [
    "Age",
    "Fare",
    "Parch",
    "Survived"
]

pair_df = eda_df[
    pair_columns
].dropna()

pair_plot = sns.pairplot(
    pair_df,
    hue="Survived"
)

pair_plot.fig.suptitle(
    "Multivariate Pairplot",
    y=1.02
)

pair_plot.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_pairplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ==========================================================
# 8. ADDITIONAL INSIGHTS
# ==========================================================

print("\n" + "=" * 60)
print("8. INSIGHT EXTRACTION")
print("=" * 60)


# ---------- Children under 10 ----------

children = eda_df[
    eda_df["Age"] < 10
]

if len(children) > 0:

    child_survival = (
        children["Survived"].mean() * 100
    )

    print(
        f"\nChildren under 10 survival rate: "
        f"{child_survival:.2f}%"
    )


# ---------- Men over 50 ----------

older_men = eda_df[
    (eda_df["Age"] > 50)
    &
    (eda_df["Sex"] == "male")
]

if len(older_men) > 0:

    older_men_survival = (
        older_men["Survived"].mean() * 100
    )

    print(
        f"Men over 50 survival rate: "
        f"{older_men_survival:.2f}%"
    )


# ---------- Median Fare ----------

median_fare = eda_df.groupby(
    "Pclass"
)["Fare"].median()

print("\nMedian Fare by Passenger Class:")
print(median_fare.round(2))


# ---------- Women 20-40 in Class 1/2 ----------

sweet_spot = eda_df[
    (eda_df["Sex"] == "female")
    &
    (eda_df["Age"].between(20, 40))
    &
    (eda_df["Pclass"].isin([1, 2]))
]

if len(sweet_spot) > 0:

    sweet_spot_survival = (
        sweet_spot["Survived"].mean() * 100
    )

    print(
        "\nWomen aged 20-40 in 1st/2nd class "
        f"survival rate: {sweet_spot_survival:.2f}%"
    )


# ==========================================================
# 9. AGE GROUP ANALYSIS
# ==========================================================

print("\n" + "=" * 60)
print("9. AGE GROUP ANALYSIS")
print("=" * 60)

eda_df["AgeGroup"] = pd.cut(
    eda_df["Age"],
    bins=[
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        100
    ],
    labels=[
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        "60+"
    ],
    include_lowest=True
)

age_survival = (
    eda_df
    .groupby(
        "AgeGroup",
        observed=False
    )["Survived"]
    .mean()
    * 100
)

print("\nSurvival Rate by Age Group:")
print(age_survival.round(2))


# ==========================================================
# 10. HYPOTHESIS TESTING
# ==========================================================

print("\n" + "=" * 60)
print("10. HYPOTHESIS TESTING")
print("=" * 60)


# H1:
# Higher-class passengers had better survival

survival_by_class = (
    eda_df
    .groupby("Pclass")["Survived"]
    .mean()
    * 100
)

h1_confirmed = (
    survival_by_class.loc[1]
    >
    survival_by_class.loc[2]
    >
    survival_by_class.loc[3]
)

print("\nH1: Higher-class passengers had better survival")

if h1_confirmed:
    print("Result: CONFIRMED")
else:
    print("Result: NOT CONFIRMED")


print("\nSurvival Rate by Class:")
print(survival_by_class.round(2))


# H2:
# Age impacted survival across age groups

print(
    "\nH2: Age impacted survival across different "
    "age groups."
)

print(
    "Result: The age-group survival rates above "
    "should be compared to evaluate the hypothesis."
)


# ==========================================================
# 11. FEATURE ENGINEERING
# ==========================================================

print("\n" + "=" * 60)
print("11. FEATURE ENGINEERING")
print("=" * 60)

eda_df["FamilySize"] = (
    eda_df["SibSp"]
    +
    eda_df["Parch"]
    +
    1
)

family_survival = (
    eda_df
    .groupby("FamilySize")["Survived"]
    .mean()
    * 100
)

print("\nSurvival Rate by Family Size:")
print(family_survival.round(2))


# ==========================================================
# 12. SAVE RESULTS
# ==========================================================

summary = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Fare Pearson Correlation",
        "Fare Spearman Correlation",
        "IQR Fare Outliers",
        "Z-Score Fare Outliers"
    ],

    "Value": [
        len(eda_df),
        len(eda_df.columns),
        round(pearson_corr, 4),
        round(spearman_corr, 4),
        len(iqr_outliers),
        len(z_outliers)
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "eda_summary.csv"
    ),
    index=False
)


class_survival.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "survival_by_class.csv"
    )
)


gender_survival.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "survival_by_gender.csv"
    )
)


age_survival.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "survival_by_age_group.csv"
    )
)

# ==========================================================
# 13. FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 60)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nAll graphs and analysis files have been "
    "saved in the outputs folder."
)