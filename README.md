# Calories Burnt Prediction System

A machine learning project that estimates calories burnt during exercise from a person's profile and workout details. It includes a training pipeline with data validation and a Streamlit web app for making predictions.

## Overview

- **Problem type:** Regression (predict `Calories`, in kcal)
- **Main model:** XGBoost Regressor (gradient boosted decision trees)
- **Baseline:** Linear Regression (to confirm XGBoost is worth the extra complexity)
- **Interface:** Streamlit web app

## Dataset

Two CSV files, merged on `User_ID` (15,000 rows):

| File | Columns |
|---|---|
| `exercise.csv` | `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp` |
| `calories.csv` | `User_ID`, `Calories` |

Observed ranges in the data:

| Feature | Min | Max |
|---|---|---|
| Age (years) | 20 | 79 |
| Height (cm) | 123 | 222 |
| Weight (kg) | 36 | 132 |
| Duration (min) | 1 | 30 |
| Heart rate (bpm) | 67 | 128 |
| Body temperature (°C) | 37.1 | 41.5 |
| Calories (kcal) | 1 | 314 |

There are no missing values and no duplicate `User_ID`s.

## Project Structure

```
.
├── app.py                         # Streamlit web app
├── train_model.py                 # Data loading, cleaning, training, evaluation
├── data/
│   ├── calories.csv
│   └── exercise.csv
├── calories_model.pkl             # Trained XGBoost model (generated)
├── calories_linear_model.pkl      # Linear fallback model (generated)
├── model_meta.json                # Feature order, encoding, ranges, metrics (generated)
└── README.md
```

## How It Works

**Pipeline in `train_model.py`:** Load → Clean → Split → Train → Evaluate → Save

1. **Load:** checks that both files and all required columns exist, then merges with `validate="one_to_one"` so duplicate IDs can't silently multiply rows.
2. **Clean:**
   - Normalises and encodes gender (male = 0, female = 1)
   - Converts numeric columns to numbers
   - Drops rows with missing or invalid values
   - Drops rows outside realistic ranges (`VALID_RANGES`)
   - Reports how many rows each step removed
   - Optional duplicate-record removal via `DROP_DUPLICATES` (off by default)
3. **Split:** train / validation / test. The test set is only used for the final score. The validation set is used for early stopping.
4. **Train:**
   - XGBoost with early stopping (the number of trees is chosen from validation data)
   - A Linear Regression baseline for comparison
   - 5-fold cross-validation on the training set
5. **Evaluate:** R², MAE and RMSE on the held-out test set. Predictions are clipped at 0 because calories can't be negative.
6. **Save:** the model, the linear fallback, and a metadata file (feature order, gender encoding, valid ranges, trained ranges, metrics).

## Results

From the latest run (30% test split, 20% of the remaining data used for validation):

| Metric | Value |
|---|---|
| Linear Regression baseline R² | 0.966 |
| XGBoost R² (test) | 99.96% |
| XGBoost MAE (test) | 0.92 kcal |
| XGBoost RMSE (test) | 1.30 kcal |
| 5-fold cross-validation R² | 0.9995 (± 0.0001) |

The average calories burnt in the data is about 89.5 kcal, so the error is roughly 1%. The test and cross-validation scores agree, which indicates the model is not overfitting to one particular split.

> **Note:** scores this close to perfect usually mean the target is almost a direct function of the inputs, as in a synthetic or very clean dataset. Real-world fitness data would be noisier and score lower. Exact numbers vary slightly with the split settings.

## Installation

Requires Python 3.9 or newer.

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit
```

## Usage

**1. Train the model** (run from the project folder, with the CSVs in `data/`):

```bash
python train_model.py
```

This prints the cleaning report and evaluation, and creates `calories_model.pkl`, `calories_linear_model.pkl` and `model_meta.json`.

**2. Start the app:**

```bash
streamlit run app.py
```

Open `http://localhost:8501` if the browser doesn't open automatically.

> Use `streamlit run app.py`, not `python app.py`. Running it with plain Python produces "missing ScriptRunContext" warnings and no web page.

## App Behaviour

The app validates every input before predicting:

- **Impossible values** (for example a heart rate of 0 or an invalid gender) show an error and no prediction.
- **Values inside the training ranges** are predicted with XGBoost (the accurate model).
- **Values outside the training ranges** (for example a 40-minute workout) use the linear model instead. The result is clearly labelled as a **rough estimate**, and the out-of-range values are listed.

`validate_input()` in `train_model.py` can be reused in any other front end.

## Limitations

- The training data has a **maximum duration of 30 minutes** (and heart rate of 128 bpm). Tree-based models can't extrapolate beyond the data they have seen, so XGBoost predictions outside these ranges would be unreliable. That is why the app switches to a labelled linear estimate there.
- The linear fallback is less accurate (about 8 kcal average error on the test split, and likely more outside the training range).
- The dataset appears to be synthetic or unusually clean, so results shouldn't be read as real-world accuracy.
- Calorie estimates are for learning and demonstration, not medical or nutritional advice.

## Possible Improvements

- Train on data with longer sessions and a wider range of heart rates
- Hyperparameter search (for example `RandomizedSearchCV`) and a larger tree limit, since early stopping reached the cap of 1000 trees
- Feature engineering (for example BMI, or Duration × Heart Rate)
- Evaluate on data from real fitness trackers
- Move shared settings (`FEATURES`, `VALID_RANGES`, `validate_input`) into a separate `config.py` so the app doesn't import the training script
