import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
from pathlib import Path

# ============================================================
# FILES
# ============================================================

# File paths

BASE_DIR = Path(__file__).resolve().parent


FILE = BASE_DIR / "urban_building_stock_datasets_17042024.csv"
MODEL_FILE = BASE_DIR / "results" / "ann_energy_model.keras"
OUTPUT = BASE_DIR / "results" / "07_validation_test_parity_plot.png"

TARGET = "Energy_Use_Intensity"
SEED = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Weather_File",
    "Building_Type",
    "Renewable_Energy_Usage",
    "Floor_Insulation_U-Value",
    "Door_Insulation_U-Value",
    "Roof_Insulation_U-Value",
    "Window_Insulation_U-Value",
    "Wall_Insulation_U-Value",
    "HVAC_Efficiency",
    "Domestic_Hot_Water_Usage",
    "Building_Orientation",
    "Lighting_Density",
    "Occupancy_Level",
    "Equipment_Density",
    "Heating_Setpoint_Temperature",
    "Heating_Setback_Temperature",
    "Air_Change_Rate",
    "Window_to_Wall_Ratio",
    "Total_Building_Area"
]

CATEGORICAL = [
    "Weather_File",
    "Building_Type",
    "Renewable_Energy_Usage"
]

NUMERICAL = [
    x for x in FEATURES
    if x not in CATEGORICAL
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    FILE,
    low_memory=False
)

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# SAME 70 / 15 / 15 SPLIT USED DURING TRAINING
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED
)


print(f"Training   : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Testing    : {len(X_test):,}")


# ============================================================
# SAME PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        NUMERICAL
    ),
    (
        "categorical",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        CATEGORICAL
    )
])

# Fit ONLY on training data
preprocessor.fit(X_train)

X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_pred = model.predict(
    X_val_scaled,
    batch_size=4096,
    verbose=0
).ravel()

val_actual = y_val.to_numpy()

val_r2 = r2_score(
    val_actual,
    val_pred
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_pred = model.predict(
    X_test_scaled,
    batch_size=4096,
    verbose=0
).ravel()

test_actual = y_test.to_numpy()

test_r2 = r2_score(
    test_actual,
    test_pred
)


# ============================================================
# SAMPLE POINTS FOR READABLE PLOT
# ============================================================

rng = np.random.default_rng(SEED)

# Plot 8,000 validation + 8,000 test samples
N = 8000

val_idx = rng.choice(
    len(val_actual),
    size=min(N, len(val_actual)),
    replace=False
)

test_idx = rng.choice(
    len(test_actual),
    size=min(N, len(test_actual)),
    replace=False
)


val_actual_plot = val_actual[val_idx]
val_pred_plot = val_pred[val_idx]

test_actual_plot = test_actual[test_idx]
test_pred_plot = test_pred[test_idx]


# ============================================================
# PERFECT PREDICTION LINE RANGE
# ============================================================

minimum = min(
    val_actual_plot.min(),
    val_pred_plot.min(),
    test_actual_plot.min(),
    test_pred_plot.min()
)

maximum = max(
    val_actual_plot.max(),
    val_pred_plot.max(),
    test_actual_plot.max(),
    test_pred_plot.max()
)


# ============================================================
# VALIDATION + TEST PARITY PLOT
# ============================================================

plt.figure(
    figsize=(8, 7)
)


# Validation
plt.scatter(
    val_actual_plot,
    val_pred_plot,
    s=12,
    alpha=0.35,
    color="red",
    label=f"Validation ($R^2$ = {val_r2:.4f})"
)


# Test
plt.scatter(
    test_actual_plot,
    test_pred_plot,
    s=12,
    alpha=0.35,
    color="blue",
    label=f"Test ($R^2$ = {test_r2:.4f})"
)


# Perfect prediction line
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="black",
    linewidth=1.5,
    label="Perfect Prediction (y = x)"
)


plt.xlabel(
    "Actual Energy Use Intensity (EUI)"
)

plt.ylabel(
    "ANN Predicted Energy Use Intensity (EUI)"
)

plt.title(
    "ANN Predicted vs. Actual Energy Use Intensity"
)

plt.legend()

plt.grid(
    alpha=0.20
)

plt.tight_layout()


# ============================================================
# SAVE + SHOW
# ============================================================

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PARITY PLOT RESULTS")
print("=" * 60)

print(
    f"Validation R² : {val_r2:.4f}"
)

print(
    f"Test R²       : {test_r2:.4f}"
)

print(
    f"\nPlot saved to:\n{OUTPUT}"
)