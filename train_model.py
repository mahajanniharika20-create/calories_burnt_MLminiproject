import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

def main():
    print("Loading data...")
    # 1. Load Datasets
    calories_df = pd.read_csv("data/calories.csv")
    exercise_df = pd.read_csv("data/exercise.csv")

    # 2. Merge Datasets on User_ID
    df = pd.merge(exercise_df, calories_df, on="User_ID")

    print(f"Dataset shape: {df.shape}")

    # 3. Data Preprocessing
    # Map Gender: Male -> 0, Female -> 1
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

    # Features and Target variable
    X = df[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
    y = df['Calories']

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Building & Training
    print("Training XGBoost Regressor model...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"R² Score: {r2 * 100:.2f}%")
    print(f"Mean Absolute Error: {mae:.2f} kcal")

    # 7. Save Model
    joblib.dump(model, "calories_model.pkl")
    print("\nModel saved successfully as 'calories_model.pkl'!")

if __name__ == "__main__":
    main()