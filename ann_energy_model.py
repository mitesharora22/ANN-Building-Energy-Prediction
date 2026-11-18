import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FILE = BASE_DIR / "urban_building_stock_datasets_17042024.csv"
OUT = BASE_DIR / "results"

OUT.mkdir(exist_ok=True)

TARGET = "Energy_Use_Intensity"
SEED = 42

os.makedirs(OUT, exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. ANN INPUT FEATURES
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


# ============================================================
# 3. CANDIDATE ANN ARCHITECTURES
# ============================================================

ARCHITECTURES = {
    "A_32": [32],
    "B_64_32": [64, 32],
    "C_64_32_16": [64, 32, 16],
    "D_64_64_64": [64, 64, 64],
    "E_128_64_32": [128, 64, 32],
    "F_128_128_64_32": [128, 128, 64, 32]
}


# ============================================================
# 4. FUNCTIONS
# ============================================================

def build_model(n_inputs, hidden_layers):

    model = Sequential()

    model.add(
        Input(shape=(n_inputs,))
    )

    for neurons in hidden_layers:

        model.add(
            Dense(
                neurons,
                activation="relu"
            )
        )

    # Regression output
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def calculate_metrics(y_true, y_pred):

    r2 = r2_score(
        y_true,
        y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return r2, mae, rmse


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("ANN BUILDING ENERGY PREDICTION")
print("=" * 75)

df = pd.read_csv(
    FILE,
    low_memory=False
)

print(f"\nObservations : {len(df):,}")
print(f"Columns      : {len(df.columns)}")
print(f"ANN features : {len(FEATURES)}")
print(f"Target       : {TARGET}")

missing = (
    df[FEATURES + [TARGET]]
    .isna()
    .sum()
    .sum()
)

duplicates = df.duplicated().sum()

print(f"Missing      : {missing:,}")
print(f"Duplicates   : {duplicates:,}")


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("DESCRIPTIVE STATISTICS")
print("=" * 75)

numeric_features = [
    feature
    for feature in FEATURES
    if pd.api.types.is_numeric_dtype(
        df[feature]
    )
]

stats = (
    df[numeric_features + [TARGET]]
    .describe()
    .T
)

print(stats)

stats.to_csv(
    os.path.join(
        OUT,
        "descriptive_statistics.csv"
    )
)


# ============================================================
# 7. PEARSON CORRELATION WITH EUI
# ============================================================

print("\n" + "=" * 75)
print("PEARSON CORRELATION WITH EUI")
print("=" * 75)

correlation = (
    df[numeric_features + [TARGET]]
    .corr(method="pearson")[TARGET]
    .drop(TARGET)
    .sort_values(
        key=abs,
        ascending=False
    )
)

print(correlation)

correlation.to_csv(
    os.path.join(
        OUT,
        "correlation_with_EUI.csv"
    ),
    header=[
        "Pearson_Correlation"
    ]
)


# Correlation figure

plt.figure(
    figsize=(9, 7)
)

correlation.sort_values().plot(
    kind="barh"
)

plt.xlabel(
    "Pearson Correlation with EUI"
)

plt.ylabel(
    "Building Feature"
)

plt.title(
    "Correlation of Building Features with EUI"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "01_correlation_EUI.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 8. INPUT AND OUTPUT
# ============================================================

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
#
# Training   = 70%
# Validation = 15%
# Testing    = 15%
#
# IMPORTANT:
# Validation chooses architecture.
# Test data is NOT used for architecture selection.
# ============================================================

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED
    )
)

X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED
    )
)


print("\n" + "=" * 75)
print("DATA SPLIT")
print("=" * 75)

print(
    f"Training   : "
    f"{len(X_train):,} (70%)"
)

print(
    f"Validation : "
    f"{len(X_val):,} (15%)"
)

print(
    f"Testing    : "
    f"{len(X_test):,} (15%)"
)


# ============================================================
# 10. PREPROCESSING
# ============================================================

NUMERICAL = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL
]


preprocessor = ColumnTransformer(
    [
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
    ]
)


# IMPORTANT:
# Fit preprocessing ONLY using training data.

X_train_scaled = (
    preprocessor
    .fit_transform(X_train)
)

X_val_scaled = (
    preprocessor
    .transform(X_val)
)

X_test_scaled = (
    preprocessor
    .transform(X_test)
)


N_INPUTS = X_train_scaled.shape[1]


print(
    f"\nOriginal features : "
    f"{len(FEATURES)}"
)

print(
    f"Encoded ANN inputs: "
    f"{N_INPUTS}"
)


# ============================================================
# 11. ARCHITECTURE COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("ANN ARCHITECTURE COMPARISON")
print("=" * 75)

architecture_results = []

trained_models = {}
training_histories = {}


for name, architecture in ARCHITECTURES.items():

    print("\n" + "-" * 75)

    print(
        f"Testing architecture: "
        f"{N_INPUTS} -> "
        + " -> ".join(
            map(str, architecture)
        )
        + " -> 1"
    )

    print("-" * 75)


    # Clear previous TensorFlow session
    tf.keras.backend.clear_session()

    # Same random seed for fair comparison
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


    model = build_model(
        N_INPUTS,
        architecture
    )


    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )


    start_time = time.time()


    history = model.fit(
        X_train_scaled,
        y_train,

        validation_data=(
            X_val_scaled,
            y_val
        ),

        epochs=100,

        batch_size=1024,

        callbacks=[
            early_stop
        ],

        verbose=0
    )


    training_time = (
        time.time()
        - start_time
    )


    # ----------------------------------------
    # Validation prediction
    # ----------------------------------------

    val_prediction = (
        model.predict(
            X_val_scaled,
            batch_size=4096,
            verbose=0
        )
        .ravel()
    )


    val_r2, val_mae, val_rmse = (
        calculate_metrics(
            y_val,
            val_prediction
        )
    )


    epochs_used = len(
        history.history["loss"]
    )


    parameters = (
        model.count_params()
    )


    architecture_text = (
        " -> ".join(
            map(str, architecture)
        )
    )


    architecture_results.append(
        {
            "Model": name,

            "Architecture":
                architecture_text,

            "Parameters":
                parameters,

            "Epochs":
                epochs_used,

            "Validation_R2":
                val_r2,

            "Validation_MAE":
                val_mae,

            "Validation_RMSE":
                val_rmse,

            "Training_Time_sec":
                training_time
        }
    )


    trained_models[name] = model

    training_histories[name] = history


    print(
        f"Parameters      : "
        f"{parameters:,}"
    )

    print(
        f"Epochs          : "
        f"{epochs_used}"
    )

    print(
        f"Validation R²   : "
        f"{val_r2:.5f}"
    )

    print(
        f"Validation MAE  : "
        f"{val_mae:.5f}"
    )

    print(
        f"Validation RMSE : "
        f"{val_rmse:.5f}"
    )

    print(
        f"Training time   : "
        f"{training_time:.1f} sec"
    )


# ============================================================
# 12. ARCHITECTURE RESULTS TABLE
# ============================================================

architecture_df = pd.DataFrame(
    architecture_results
)


architecture_df = (
    architecture_df
    .sort_values(
        "Validation_RMSE"
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 75)
print("ARCHITECTURE COMPARISON RESULTS")
print("=" * 75)

print(
    architecture_df
    .to_string(index=False)
)


architecture_df.to_csv(
    os.path.join(
        OUT,
        "architecture_comparison.csv"
    ),
    index=False
)


# ============================================================
# 13. SELECT BEST ARCHITECTURE
#
# Selection criterion:
# LOWEST VALIDATION RMSE
# ============================================================

BEST_NAME = (
    architecture_df
    .iloc[0]["Model"]
)

BEST_ARCHITECTURE = (
    ARCHITECTURES[
        BEST_NAME
    ]
)


print("\n" + "=" * 75)
print("SELECTED ANN ARCHITECTURE")
print("=" * 75)

print(
    "Selected model:",
    BEST_NAME
)

print(
    "Hidden layers:",
    BEST_ARCHITECTURE
)

print(
    "Reason:"
    " Lowest validation RMSE"
)


# Retrieve selected trained model

final_model = (
    trained_models[
        BEST_NAME
    ]
)

final_history = (
    training_histories[
        BEST_NAME
    ]
)


# ============================================================
# 14. ARCHITECTURE COMPARISON FIGURE
# ============================================================

plot_arch = (
    architecture_df
    .sort_values(
        "Validation_RMSE",
        ascending=True
    )
)


plt.figure(
    figsize=(9, 6)
)

plt.barh(
    plot_arch["Architecture"],
    plot_arch["Validation_RMSE"]
)

plt.xlabel(
    "Validation RMSE"
)

plt.ylabel(
    "ANN Hidden-Layer Architecture"
)

plt.title(
    "ANN Architecture Comparison"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "02_architecture_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. TRAINING HISTORY OF SELECTED MODEL
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    final_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    final_history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Selected ANN Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "03_training_validation_loss.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 16. FINAL TEST SET EVALUATION
#
# Test set is used ONLY AFTER architecture selection.
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST SET EVALUATION")
print("=" * 75)


test_prediction = (
    final_model.predict(
        X_test_scaled,
        batch_size=4096,
        verbose=0
    )
    .ravel()
)


r2, mae, rmse = (
    calculate_metrics(
        y_test,
        test_prediction
    )
)


# MAPE:
# Ignore values close to zero because percentage
# error becomes unstable.

actual_test = (
    y_test.values
)

mape_mask = (
    np.abs(actual_test) >= 1
)

mape = (
    np.mean(
        np.abs(
            (
                actual_test[
                    mape_mask
                ]
                -
                test_prediction[
                    mape_mask
                ]
            )
            /
            actual_test[
                mape_mask
            ]
        )
    )
    * 100
)


print(
    f"R²   = {r2:.4f}"
)

print(
    f"MAE  = {mae:.4f}"
)

print(
    f"RMSE = {rmse:.4f}"
)

print(
    f"MAPE = {mape:.2f}%"
)


performance = pd.DataFrame(
    {
        "Metric": [
            "R2",
            "MAE",
            "RMSE",
            "MAPE_percent"
        ],

        "Value": [
            r2,
            mae,
            rmse,
            mape
        ]
    }
)


performance.to_csv(
    os.path.join(
        OUT,
        "final_test_performance.csv"
    ),
    index=False
)


# ============================================================
# 17. ACTUAL VS PREDICTED FIGURE
# ============================================================

rng = np.random.default_rng(
    SEED
)

n_plot = min(
    20000,
    len(y_test)
)


plot_indices = rng.choice(
    len(y_test),
    n_plot,
    replace=False
)


actual_plot = (
    actual_test[
        plot_indices
    ]
)

prediction_plot = (
    test_prediction[
        plot_indices
    ]
)


plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    actual_plot,
    prediction_plot,
    s=8,
    alpha=0.25
)


low = min(
    actual_plot.min(),
    prediction_plot.min()
)

high = max(
    actual_plot.max(),
    prediction_plot.max()
)


plt.plot(
    [low, high],
    [low, high],
    "--"
)


plt.xlabel(
    "Actual EUI"
)

plt.ylabel(
    "Predicted EUI"
)

plt.title(
    f"Actual vs. Predicted Energy Use Intensity\n"
    f"R² = {r2:.4f}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "04_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 18. RESIDUAL STATISTICS
# ============================================================

residuals = (
    actual_test
    -
    test_prediction
)


residual_stats = (
    pd.Series(
        residuals
    )
    .describe()
)


print("\n" + "=" * 75)
print("RESIDUAL STATISTICS")
print("=" * 75)

print(
    residual_stats
)


residual_stats.to_csv(
    os.path.join(
        OUT,
        "residual_statistics.csv"
    )
)


# ============================================================
# 19. RESIDUAL FIGURE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    residuals,
    bins=80
)

plt.xlabel(
    "Residual "
    "(Actual EUI - Predicted EUI)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of ANN Prediction Errors"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "05_residual_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 20. TEST CASE DEMONSTRATION
# ============================================================

print("\n" + "=" * 75)
print("ANN TEST CASE")
print("=" * 75)


test_case = (
    X_test.iloc[[0]]
    .copy()
)


actual_eui = float(
    y_test.iloc[0]
)


processed_case = (
    preprocessor.transform(
        test_case
    )
)


predicted_eui = float(
    final_model.predict(
        processed_case,
        verbose=0
    )[0, 0]
)


absolute_error = abs(
    actual_eui
    -
    predicted_eui
)


if abs(actual_eui) > 1e-8:

    percent_error = (
        absolute_error
        /
        abs(actual_eui)
        *
        100
    )

else:

    percent_error = np.nan


print(
    test_case.to_string(
        index=False
    )
)


print(
    f"\nActual EUI    : "
    f"{actual_eui:.3f}"
)

print(
    f"Predicted EUI : "
    f"{predicted_eui:.3f}"
)

print(
    f"Absolute Error: "
    f"{absolute_error:.3f}"
)

print(
    f"Percent Error : "
    f"{percent_error:.2f}%"
)


# Save test case

test_case_output = (
    test_case.copy()
)

test_case_output[
    "Actual_EUI"
] = actual_eui

test_case_output[
    "Predicted_EUI"
] = predicted_eui

test_case_output[
    "Absolute_Error"
] = absolute_error

test_case_output[
    "Percent_Error"
] = percent_error


test_case_output.to_csv(
    os.path.join(
        OUT,
        "ANN_test_case.csv"
    ),
    index=False
)


# ============================================================
# 21. TEST CASE FIGURE
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.bar(
    [
        "Actual EUI",
        "ANN Predicted EUI"
    ],
    [
        actual_eui,
        predicted_eui
    ]
)

plt.ylabel(
    "Energy Use Intensity"
)

plt.title(
    "ANN Test Case"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "06_test_case.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

final_model.save(
    os.path.join(
        OUT,
        "ann_energy_model.keras"
    )
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

architecture_string = (
    f"{N_INPUTS} -> "
    +
    " -> ".join(
        map(
            str,
            BEST_ARCHITECTURE
        )
    )
    +
    " -> 1"
)


print("\n" + "=" * 75)
print("ANN ANALYSIS COMPLETE")
print("=" * 75)


print(
    f"""
DATASET
-------
Observations        : {len(df):,}
ANN original features: {len(FEATURES)}
ANN encoded inputs  : {N_INPUTS}

DATA SPLIT
----------
Training            : {len(X_train):,}
Validation          : {len(X_val):,}
Testing             : {len(X_test):,}

ARCHITECTURE SELECTION
----------------------
Architectures tested: {len(ARCHITECTURES)}
Selected model      : {BEST_NAME}
Selected architecture:
{architecture_string}

Selection criterion:
Lowest validation RMSE

FINAL TEST PERFORMANCE
----------------------
R²   = {r2:.4f}
MAE  = {mae:.4f}
RMSE = {rmse:.4f}
MAPE = {mape:.2f}%

TEST CASE
---------
Actual EUI    = {actual_eui:.3f}
Predicted EUI = {predicted_eui:.3f}
Absolute Error= {absolute_error:.3f}
Percent Error = {percent_error:.2f}%

RESULTS SAVED TO
----------------
{OUT}
"""
)