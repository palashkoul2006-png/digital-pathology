import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset

data = pd.read_csv("menstrual_cycle_dataset_with_factors.csv")

print("Columns:", data.columns)

# Select only needed columns

data = data[[
"Age",
"BMI",
"Stress Level",
"Exercise Frequency",
"Sleep Hours",
"Cycle Length",
"Period Length"
]]

# Convert Exercise Frequency to numeric

data["Exercise Frequency"] = data["Exercise Frequency"].replace({
"Low": 1,
"Moderate": 2,
"High": 3
})

# Remove any rows that still have text / missing values

data = data.dropna()

# Create label manually

def create_label(row):
    if row["Cycle Length"] < 21 or row["Cycle Length"] > 35:
        return 2   # irregular
    elif row["Period Length"] > 7:
        return 1   # warning
    else:
        return 0   # healthy

data["label"] = data.apply(create_label, axis=1)

# Features and target

X = data.drop("label", axis=1)
y = data["label"]

print("Sample data:\n", X.head())   # DEBUG

# Train model

model = RandomForestClassifier()
model.fit(X, y)

# Save model

joblib.dump(model, "model.pkl")

print("✅ Model trained successfully!")
