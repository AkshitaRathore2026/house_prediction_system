# ==============================
# HOUSE PRICE PREDICTION MODEL
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)






# ==============================
# STEP 1: LOAD DATASET
# ==============================

df = pd.read_csv("data/housing.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())







# ==============================
# STEP 2: HANDLE MISSING VALUES
# ==============================

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])






# ==============================
# STEP 3: ENCODE CATEGORICAL DATA
# ==============================

categorical_cols = df.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        )
    ],
    remainder='passthrough'
)






# ==============================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ==============================

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(
            "assets/correlation_heatmap.png",
            bbox_inches='tight'
        )
plt.show()

# Distribution Plot
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig(
            "assets/distribution_plot.png",
            bbox_inches='tight'
        )
plt.show()

# Boxplot for Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Outlier Detection")
plt.savefig(
            "assets/outlier_boxplot.png",
            bbox_inches='tight'
        )
plt.show()






# ==============================
# STEP 5: FEATURE ENGINEERING
# ==============================

if 'area' in df.columns and 'bedrooms' in df.columns:
    df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)

df['total_rooms'] = (df['bedrooms'] + df['bathrooms'])

df['luxury_score'] = (
                df['airconditioning'].map({'yes':1, 'no':0}) +
                df['hotwaterheating'].map({'yes':1, 'no':0}) +
                df['guestroom'].map({'yes':1, 'no':0})
            )

df['parking_per_bedroom'] = (df['parking'] / (df['bedrooms'] + 1))

df['area_per_story'] = (df['area'] / (df['stories'] + 1))






# ==============================
# REMOVE OUTLIERS
# ==============================

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[
    ~(
        (df['price'] < (Q1 - 1.5 * IQR)) |
        (df['price'] > (Q3 + 1.5 * IQR))
    )
]






# ==============================
# STEP 6: DEFINE FEATURES & TARGET
# ==============================

# Replace 'price' with your target column
X = df.drop("price", axis=1)
y = np.log1p(df["price"])






# ==============================
# STEP 7: TRAIN TEST SPLIT
# ==============================

print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)






# ==============================
# STEP 8: MODEL TRAINING
# ==============================

# Random Forest Pipeline
rf_model = Pipeline([
    ('preprocessing', preprocessor),
#    ('scaling', StandardScaler(with_mean=False)),
    ('model', RandomForestRegressor(
                            n_estimators=500,
                            max_depth=30,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            random_state=42
                        ))
                    ])
rf_model.fit(X_train, y_train)



# XGBoost Pipeline
xgb_model = Pipeline([
    ('preprocessing', preprocessor),
#    ('scaling', StandardScaler(with_mean=False)),
    ('model', XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.03,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    ))
                ])
xgb_model.fit(X_train, y_train)



# Ridge Pipeline
ridge_model = Pipeline([
    ('preprocessing', preprocessor),
    ('scaling', StandardScaler(with_mean=False)),
    ('model', Ridge(alpha=1.0))
])
ridge_model.fit(X_train, y_train)



# Lasso Pipeline
lasso_model = Pipeline([
    ('preprocessing', preprocessor),
    ('scaling', StandardScaler(with_mean=False)),
    ('model', Lasso(alpha=0.0001))
])
lasso_model.fit(X_train, y_train)






# ==============================
# STEP 9: MODEL EVALUATION FUNCTION
# ==============================

def evaluate_model(model, model_name):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n===== {model_name} =====")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)




    # =========================
    # ACTUAL VS PREDICTED GRAPH
    # =========================

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, predictions, color='blue', alpha=0.6, label='Predicted Values')
    plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linestyle='--',
    linewidth=2,
    label='Actual Values'
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.legend()
    plt.savefig(
                f"assets/{model_name}_prediction_graph.png",
                bbox_inches='tight'
            )
    plt.show()
    return r2



# Evaluate All Models
rf_r2 = evaluate_model(rf_model, "Random Forest")
xgb_r2 = evaluate_model(xgb_model, "XGBoost")
ridge_r2 = evaluate_model(ridge_model, "Ridge Regression")
lasso_r2 = evaluate_model(lasso_model, "Lasso Regression")



# Dictionary of models
model_scores = {
    "Random Forest": rf_r2,
    "XGBoost": xgb_r2,
    "Ridge Regression": ridge_r2,
    "Lasso Regression": lasso_r2
}



# Find best model
models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Ridge Regression": ridge_model,
    "Lasso Regression": lasso_model
}
best_model_name = max(
    model_scores,
    key=model_scores.get
)
best_model = models[best_model_name]
print("\nBest Model:", best_model_name)






# ==============================
# STEP 11: SAVE MODEL
# ==============================

# Save best model name
joblib.dump(
    best_model_name,
    "models/best_model_name.pkl"
)

# Save all model scores
joblib.dump(
    model_scores,
    "models/model_scores.pkl"
)

joblib.dump(best_model, "models/house_price_model.pkl")

print("\nModel Saved Successfully!")
