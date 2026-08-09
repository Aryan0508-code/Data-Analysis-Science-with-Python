import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.power import TTestIndPower

warnings.filterwarnings("ignore")


# ============================================================
# TASK 4: A/B TESTING AND HYPOTHESIS TESTING
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "ab_test_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("TASK 4: A/B TESTING AND HYPOTHESIS TESTING")
print("=" * 65)

ab_data = pd.read_csv(DATA_FILE)

print("\nSample Data:")
print(ab_data.head())

print("\nDataset Shape:")
print(ab_data.shape)

print("\nColumns:")
print(ab_data.columns.tolist())

print("\nMissing Values:")
print(ab_data.isnull().sum())


# ============================================================
# 2. BASIC GROUP SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("2. GROUP SUMMARY")
print("=" * 65)

group_summary = (
    ab_data
    .groupby("group")
    .agg(
        visitors=("converted", "count"),
        conversions=("converted", "sum")
    )
)

group_summary["conversion_rate"] = (
    group_summary["conversions"]
    /
    group_summary["visitors"]
)

group_summary["conversion_rate_percent"] = (
    group_summary["conversion_rate"] * 100
)

print("\nGroup Summary:")
print(group_summary)


# ============================================================
# 3. FORMULATE HYPOTHESES
# ============================================================

print("\n" + "=" * 65)
print("3. HYPOTHESIS FORMULATION")
print("=" * 65)

print("""
H0: New design conversion rate <= Old design
    (p_new - p_old <= 0)

H1: New design conversion rate > Old design
    (p_new - p_old > 0)
""")


# ============================================================
# 4. EXTRACT CONTROL AND TREATMENT DATA
# ============================================================

control = ab_data[
    ab_data["group"].str.lower() == "control"
]

treatment = ab_data[
    ab_data["group"].str.lower() == "treatment"
]

conv_old = control["converted"].sum()
conv_new = treatment["converted"].sum()

n_old = len(control)
n_new = len(treatment)

rate_old = conv_old / n_old
rate_new = conv_new / n_new

absolute_difference = (
    rate_new - rate_old
)

relative_lift = (
    absolute_difference / rate_old
) * 100

print("\nControl Visitors:", n_old)
print("Control Conversions:", conv_old)
print(
    f"Control Conversion Rate: "
    f"{rate_old * 100:.2f}%"
)

print("\nTreatment Visitors:", n_new)
print("Treatment Conversions:", conv_new)
print(
    f"Treatment Conversion Rate: "
    f"{rate_new * 100:.2f}%"
)

print(
    f"\nAbsolute Conversion Difference: "
    f"{absolute_difference * 100:.2f} percentage points"
)

print(
    f"Relative Conversion Lift: "
    f"{relative_lift:.2f}%"
)


# ============================================================
# 5. TWO-PROPORTION Z-TEST
# ============================================================

print("\n" + "=" * 65)
print("5. TWO-PROPORTION Z-TEST")
print("=" * 65)

counts = np.array([
    conv_new,
    conv_old
])

sample_sizes = np.array([
    n_new,
    n_old
])

z_score, p_value = proportions_ztest(
    counts,
    sample_sizes,
    alternative="larger"
)

print(
    f"Z-score: {z_score:.4f}"
)

print(
    f"P-value: {p_value:.6f}"
)

alpha = 0.05

if p_value < alpha:
    print(
        "\nDecision: Reject H0"
    )

    print(
        "Conclusion: The treatment conversion "
        "rate is statistically significantly "
        "higher than the control conversion rate."
    )

else:
    print(
        "\nDecision: Fail to reject H0"
    )

    print(
        "Conclusion: There is not enough "
        "evidence to conclude that treatment "
        "has a higher conversion rate."
    )


# ============================================================
# 6. 95% CONFIDENCE INTERVALS
# ============================================================

print("\n" + "=" * 65)
print("6. 95% CONFIDENCE INTERVALS")
print("=" * 65)

ci_old = proportion_confint(
    conv_old,
    n_old,
    alpha=0.05,
    method="normal"
)

ci_new = proportion_confint(
    conv_new,
    n_new,
    alpha=0.05,
    method="normal"
)

print(
    f"Control 95% CI: "
    f"{ci_old[0] * 100:.2f}% - "
    f"{ci_old[1] * 100:.2f}%"
)

print(
    f"Treatment 95% CI: "
    f"{ci_new[0] * 100:.2f}% - "
    f"{ci_new[1] * 100:.2f}%"
)


# ============================================================
# 7. CONFIDENCE INTERVAL VISUALIZATION
# ============================================================

rates = [
    rate_old,
    rate_new
]

lower_errors = [
    rate_old - ci_old[0],
    rate_new - ci_new[0]
]

upper_errors = [
    ci_old[1] - rate_old,
    ci_new[1] - rate_new
]

plt.figure(figsize=(10, 6))

plt.errorbar(
    [0, 1],
    rates,
    yerr=[
        lower_errors,
        upper_errors
    ],
    fmt="o",
    capsize=10
)

plt.xticks(
    [0, 1],
    ["Control", "Treatment"]
)

plt.ylabel("Conversion Rate")

plt.title(
    "95% Confidence Intervals for Conversion Rates"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_confidence_intervals.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. CHI-SQUARE TEST
# ============================================================

print("\n" + "=" * 65)
print("8. CHI-SQUARE TEST: DEVICE VS CONVERSION")
print("=" * 65)

if "device" in ab_data.columns:

    contingency_table = pd.crosstab(
        ab_data["device"],
        ab_data["converted"]
    )

    print("\nContingency Table:")
    print(contingency_table)

    chi2, chi_p, dof, expected = (
        stats.chi2_contingency(
            contingency_table
        )
    )

    print(
        f"\nChi-square statistic: "
        f"{chi2:.4f}"
    )

    print(
        f"Chi-square p-value: "
        f"{chi_p:.5f}"
    )

    if chi_p < alpha:
        print(
            "Result: Significant relationship "
            "between device type and conversion."
        )
    else:
        print(
            "Result: No statistically significant "
            "relationship between device type "
            "and conversion."
        )

else:

    print(
        "Device column is not available "
        "in the dataset."
    )


# ============================================================
# 9. T-TEST: SESSION DURATION
# ============================================================

print("\n" + "=" * 65)
print("9. T-TEST: SESSION DURATION")
print("=" * 65)

if "session_duration" in ab_data.columns:

    duration_control = control[
        "session_duration"
    ].dropna()

    duration_treatment = treatment[
        "session_duration"
    ].dropna()

    t_stat, t_p_value = stats.ttest_ind(
        duration_treatment,
        duration_control,
        equal_var=False
    )

    print(
        f"T-statistic: {t_stat:.4f}"
    )

    print(
        f"T-test p-value: "
        f"{t_p_value:.4f}"
    )

    if t_p_value < alpha:
        print(
            "Result: Significant difference "
            "in session duration."
        )
    else:
        print(
            "Result: No statistically significant "
            "difference in session duration."
        )

else:

    print(
        "session_duration column is not "
        "available in the dataset."
    )


# ============================================================
# 10. POWER ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("10. POWER ANALYSIS")
print("=" * 65)

effect_size = 0.2
power = 0.8
alpha_power = 0.05

analysis = TTestIndPower()

sample_size = analysis.solve_power(
    effect_size=effect_size,
    power=power,
    alpha=alpha_power,
    alternative="two-sided"
)

print(
    f"Required sample size per group: "
    f"{int(np.ceil(sample_size))}"
)


# ============================================================
# 11. RANDOMIZATION / DEVICE DISTRIBUTION CHECK
# ============================================================

print("\n" + "=" * 65)
print("11. RANDOMIZATION CHECK")
print("=" * 65)

if "device" in ab_data.columns:

    device_distribution = pd.crosstab(
        ab_data["group"],
        ab_data["device"],
        normalize="index"
    ) * 100

    print(
        "\nDevice distribution by group (%):"
    )

    print(
        device_distribution.round(2)
    )


# ============================================================
# 12. BUSINESS IMPACT
# ============================================================

print("\n" + "=" * 65)
print("12. BUSINESS IMPACT")
print("=" * 65)

print(
    f"\nControl conversion rate: "
    f"{rate_old * 100:.2f}%"
)

print(
    f"Treatment conversion rate: "
    f"{rate_new * 100:.2f}%"
)

print(
    f"Absolute improvement: "
    f"{absolute_difference * 100:.2f} percentage points"
)

print(
    f"Relative lift: "
    f"{relative_lift:.2f}%"
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Control Visitors",
        "Control Conversions",
        "Control Conversion Rate",
        "Treatment Visitors",
        "Treatment Conversions",
        "Treatment Conversion Rate",
        "Absolute Difference",
        "Relative Lift",
        "Z Score",
        "P Value",
        "Control CI Lower",
        "Control CI Upper",
        "Treatment CI Lower",
        "Treatment CI Upper"
    ],

    "Value": [
        n_old,
        conv_old,
        rate_old,
        n_new,
        conv_new,
        rate_new,
        absolute_difference,
        relative_lift,
        z_score,
        p_value,
        ci_old[0],
        ci_old[1],
        ci_new[0],
        ci_new[1]
    ]
})

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ab_test_results.csv"
    ),
    index=False
)


# ============================================================
# 14. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 65)
print("FINAL CONCLUSION")
print("=" * 65)

if p_value < alpha:

    print(
        "The A/B test provides statistically "
        "significant evidence that the treatment "
        "design improves conversion rate."
    )

    print(
        "Recommendation: Consider rolling out "
        "the treatment design while continuing "
        "to monitor performance."
    )

else:

    print(
        "The A/B test does not provide sufficient "
        "statistical evidence that the treatment "
        "design improves conversion rate."
    )

    print(
        "Recommendation: Continue testing or "
        "collect additional evidence before "
        "a full rollout."
    )

print(
    "\nTask 4 completed successfully."
)

print(
    "Results and visualization saved in "
    "the outputs folder."
)