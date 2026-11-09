import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

RANDOM_STATE=42
DATA_DIR = Path("data")
MODEL_PATH = Path("calories_model.pkl")
LINEAR_PATH = Path("calories_linear_model.pkl")  # rough fallback outside training ranges
META_PATH = Path("model_meta.json")

DROP_DUPLICATES = False  # identical rows with different User_IDs can be legitimate
GENDER_MAP = {"male": 0, "female": 1}
FEATURES = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
TARGET = "Calories"

# Plausible (min, max) ranges. Rows outside these are treated as bad data.
VALID_RANGES = {
    "Age": (10, 100),
    "Height": (100, 230),       # cm
    "Weight": (25, 250),        # kg
    "Duration": (1, 300),       # minutes
    "Heart_Rate": (40, 220),    # bpm
    "Body_Temp": (35, 42),      # deg C
    "Calories": (0, 5000),      # kcal
}



def load_data() -> pd.DataFrame:
    """Load both CSVs, check they exist and have the expected columns, then merge."""
    paths = {"calories": DATA_DIR / "calories.csv", "exercise": DATA_DIR / "exercise.csv"}
    for name, p in paths.items():
        if not p.exists():
            sys.exit(f"ERROR: {name} file not found at '{p}'")

    calories_df = pd.read_csv(paths["calories"])
    exercise_df = pd.read_csv(paths["exercise"])

    needed_cal = {"User_ID", TARGET}
    needed_ex = {"User_ID", "Gender", "Age", "Height", "Weight",
                 "Duration", "Heart_Rate", "Body_Temp"}
    for df, needed, name in [(calories_df, needed_cal, "calories.csv"),
                             (exercise_df, needed_ex, "exercise.csv")]:
        missing = needed - set(df.columns)
        if missing:
            sys.exit(f"ERROR: {name} is missing columns: {sorted(missing)}")

    # Duplicate IDs would multiply rows in the merge, so drop them first and report.
    for df, name in [(calories_df, "calories.csv"), (exercise_df, "exercise.csv")]:
        dups = df.duplicated(subset="User_ID").sum()
        if dups:
            print(f"WARNING: {dups} duplicate User_ID rows in {name}; keeping first")
    calories_df = calories_df.drop_duplicates(subset="User_ID", keep="first")
    exercise_df = exercise_df.drop_duplicates(subset="User_ID", keep="first")

    df = pd.merge(exercise_df, calories_df, on="User_ID", how="inner",
                  validate="one_to_one")

    lost = len(exercise_df) - len(df)
    if lost:
        print(f"WARNING: {lost} exercise rows had no matching calories row")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise types, drop missing / duplicate / implausible rows, and report each step."""
    n0 = len(df)
    df = df.copy()

    # Gender: normalise text, map, and make sure nothing unmapped slips through.
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map(GENDER_MAP)

    # Force numeric columns to numbers; bad strings become NaN and are dropped below.
    numeric_cols = [c for c in FEATURES + [TARGET] if c != "Gender"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing values
    missing = df[FEATURES + [TARGET]].isna().sum()
    if missing.any():
        print("Missing values found:\n", missing[missing > 0].to_string())
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Dropped for missing/invalid values: {n0 - len(df)}")

    # Exact duplicate records (ignoring User_ID) -- optional, see DROP_DUPLICATES
    n_dup = int(df.duplicated(subset=FEATURES + [TARGET]).sum())
    if DROP_DUPLICATES:
        df = df.drop_duplicates(subset=FEATURES + [TARGET])
        print(f"Dropped exact duplicate records: {n_dup}")
    elif n_dup:
        print(f"Note: {n_dup} identical records with different User_IDs kept")

    # Plausibility ranges
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        bad = ~df[col].between(lo, hi)
        if bad.any():
            print(f"  {col}: {bad.sum()} rows outside [{lo}, {hi}]")
        mask &= ~bad
    df = df[mask]
    print(f"Dropped for out-of-range values: {before - len(df)}")

    if len(df) < 100:
        sys.exit(f"ERROR: only {len(df)} rows left after cleaning; check your data")

    print(f"Rows kept: {len(df)} of {n0}")
    return df.reset_index(drop=True)


def validate_input(row: dict, meta_path: Path = META_PATH) -> list:
    """Call this in your prediction app before predicting.

    `row` uses the same keys as FEATURES, with Gender as 'male' or 'female'.
    Raises ValueError for missing/invalid fields. Returns a list of warnings for
    values outside the range the model was trained on (tree models cannot
    extrapolate, so predictions there are unreliable).
    """
    missing = set(FEATURES) - set(row)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if str(row["Gender"]).strip().lower() not in GENDER_MAP:
        raise ValueError("Gender must be 'male' or 'female'")

    for col, (lo, hi) in VALID_RANGES.items():
        if col in row and not (lo <= float(row[col]) <= hi):
            raise ValueError(f"{col} must be between {lo} and {hi}")

    warnings = []
    if Path(meta_path).exists():
        trained = json.loads(Path(meta_path).read_text()).get("trained_ranges", {})
        for col, (lo, hi) in trained.items():
            if col in row and not (lo <= float(row[col]) <= hi):
                warnings.append(
                    f"{col}={row[col]} is outside the training range [{lo}, {hi}]; "
                    "prediction may be unreliable"
                )
    return warnings


def main():
    print("Loading data...")
    df = load_data()
    print(f"Merged shape: {df.shape}")

    print("\nCleaning data...")
    df = clean_data(df)

    X = df[FEATURES]
    y = df[TARGET]

    # Train / test split, then a validation split carved from train for early stopping
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE
    )

    # Baseline to confirm XGBoost is actually adding value
    baseline = LinearRegression().fit(X_train, y_train)
    base_r2 = r2_score(y_test, baseline.predict(X_test))
    print(f"\nBaseline (LinearRegression) R²: {base_r2:.4f}")

    # XGBoost with early stopping
    print("Training XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best_n = model.best_iteration + 1
    print(f"Best number of trees: {best_n}")

    # Cross-validation on the training set with the chosen tree count
    cv_model = XGBRegressor(
        n_estimators=best_n, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    cv_scores = cross_val_score(
        cv_model, X_train, y_train,
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2",
    )
    print(f"5-fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Final evaluation on held-out test set (predictions clipped, calories can't be < 0)
    y_pred = np.clip(model.predict(X_test), 0, None)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Test Set Evaluation ---")
    print(f"R² Score: {r2 * 100:.2f}%")
    print(f"MAE:      {mae:.2f} kcal")
    print(f"RMSE:     {rmse:.2f} kcal")

    # Save the model 
    joblib.dump(model, MODEL_PATH)

    #  linear baseline 
    joblib.dump(baseline, LINEAR_PATH)

    # Save metadata 
    meta = {
        "features": FEATURES,
        "gender_map": GENDER_MAP,
        "valid_ranges": VALID_RANGES,
        "trained_ranges": {
            c: [float(df[c].min()), float(df[c].max())]
            for c in FEATURES if c != "Gender"
        },
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse,
                    "cv_r2_mean": float(cv_scores.mean()),
                    "baseline_r2": base_r2},
        "n_trees": best_n,
        "train_rows": len(X_train),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved '{MODEL_PATH}', '{LINEAR_PATH}' and '{META_PATH}'")


if __name__ == "__main__":
    main()
