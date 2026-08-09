import pandas as pd
import numpy as np




INPUT_FILE = "../data/messy_sales_data.csv"
OUTPUT_FILE = "../data/cleaned_sales_data.csv"



print("=" * 60)
print("1. DATA INGESTION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nBasic statistics:")
print(df.describe(include="all"))




print("\n" + "=" * 60)
print("2. DUPLICATE REMOVAL")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate rows found:", duplicate_count)

df = df.drop_duplicates()

print("Duplicate rows after cleaning:", df.duplicated().sum())




print("\n" + "=" * 60)
print("3. COLUMN MANAGEMENT")
print("=" * 60)

print("Original columns:")
print(df.columns.tolist())


df = df.drop(columns=["notes", "temp_id"])


df = df.rename(columns={
    "customer_id": "customer_id",
    "sale_date": "sale_date",
    "price": "price",
    "revenue": "revenue",
    "quantity": "quantity"
})

print("\nColumns after management:")
print(df.columns.tolist())




print("\n" + "=" * 60)
print("4. MISSING VALUE HANDLING")
print("=" * 60)

print("Missing values before cleaning:")
print(df.isna().sum())

# Remove rows where customer_id is missing
df = df.dropna(subset=["customer_id"])

# Fill missing numerical revenue with median
revenue_median = df["revenue"].median()

df["revenue"] = df["revenue"].fillna(revenue_median)

print("\nMissing values after cleaning:")
print(df.isna().sum())


# --------------------------------------------
# 5. DATA TYPE CORRECTION
# --------------------------------------------

print("\n" + "=" * 60)
print("5. DATA TYPE CORRECTION")
print("=" * 60)

# Convert sale_date to datetime
df["sale_date"] = pd.to_datetime(
    df["sale_date"],
    errors="coerce"
)

# Remove $ and convert price to numeric
df["price"] = (
    df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# Convert quantity and revenue to numeric
df["quantity"] = pd.to_numeric(
    df["quantity"],
    errors="coerce"
)

df["revenue"] = pd.to_numeric(
    df["revenue"],
    errors="coerce"
)

print("\nData types after correction:")
print(df.dtypes)


# --------------------------------------------
# 6. FORMAT STANDARDIZATION
# --------------------------------------------

print("\n" + "=" * 60)
print("6. FORMAT STANDARDIZATION")
print("=" * 60)

# Standardize product names
df["product"] = (
    df["product"]
    .str.lower()
    .str.strip()
)

# Standardize regions
df["region"] = (
    df["region"]
    .str.strip()
    .str.lower()
)

df["region"] = df["region"].replace({
    "west": "Western",
    "south": "Southern"
})

# Standardize product names
df["product"] = df["product"].replace({
    "laptop": "Laptop",
    "phone": "Phone",
    "headphones": "Headphones",
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "monitor": "Monitor"
})

print("\nUnique products:")
print(df["product"].unique())

print("\nUnique regions:")
print(df["region"].unique())


# --------------------------------------------
# HANDLE INVALID DATES
# --------------------------------------------

print("\nInvalid dates converted to NaT:")
print(df[df["sale_date"].isna()])


# --------------------------------------------
# FINAL CLEANING
# --------------------------------------------

# Remove rows where date could not be converted
df = df.dropna(subset=["sale_date"])

# Reset index
df = df.reset_index(drop=True)


# --------------------------------------------
# 7. VALIDATION
# --------------------------------------------

print("\n" + "=" * 60)
print("7. DATA VALIDATION")
print("=" * 60)

print("\nFinal shape:")
print(df.shape)

print("\nFinal missing values:")
print(df.isna().sum())

print("\nFinal duplicate count:")
print(df.duplicated().sum())

print("\nFinal data types:")
print(df.dtypes)


# Assertions
assert df.duplicated().sum() == 0
assert df["customer_id"].isna().sum() == 0
assert df["sale_date"].isna().sum() == 0


# --------------------------------------------
# SAVE CLEAN DATASET
# --------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("TASK COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")

print("\nFinal cleaned dataset:")
print(df)