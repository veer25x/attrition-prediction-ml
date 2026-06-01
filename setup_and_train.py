"""
Complete HR Attrition Dataset Setup and Model Training
This script creates a realistic dataset and trains the XGBoost model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import pickle
import os

print("=" * 70)
print(" HR ATTRITION PREDICTION - COMPLETE SETUP")
print("=" * 70)

# Step 1: Create Realistic Dataset
print("\n📊 Step 1: Creating HR Attrition Dataset...")

np.random.seed(42)
n_samples = 1470  # Same size as IBM dataset

# Generate realistic employee data
data = {
    'Age': np.random.normal(36, 10, n_samples).clip(18, 60).astype(int),
    'DailyRate': np.random.uniform(500, 1500, n_samples).astype(int),
    'DistanceFromHome': np.random.exponential(5, n_samples).clip(1, 30).astype(int),
    'Education': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    'EnvironmentSatisfaction': np.random.choice([1, 2, 3, 4], n_samples, p=[0.12, 0.18, 0.35, 0.35]),
    'HourlyRate': np.random.uniform(30, 100, n_samples).astype(int),
    'JobInvolvement': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.45, 0.25]),
    'JobLevel': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.3, 0.32, 0.2, 0.12, 0.06]),
    'JobSatisfaction': np.random.choice([1, 2, 3, 4], n_samples, p=[0.12, 0.18, 0.35, 0.35]),
    'MonthlyIncome': np.random.lognormal(8.5, 0.5, n_samples).astype(int).clip(2000, 20000),
    'MonthlyRate': np.random.uniform(8000, 25000, n_samples).astype(int),
    'NumCompaniesWorked': np.random.poisson(2, n_samples).clip(0, 10),
    'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
    'PercentSalaryHike': np.random.uniform(10, 25, n_samples).astype(int),
    'PerformanceRating': np.random.choice([1, 2, 3, 4], n_samples, p=[0.02, 0.05, 0.68, 0.25]),
    'RelationshipSatisfaction': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.18, 0.4, 0.32]),
    'StockOptionLevel': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'TotalWorkingYears': np.random.normal(12, 8, n_samples).clip(0, 40).astype(int),
    'TrainingTimesLastYear': np.random.poisson(2, n_samples).clip(0, 6),
    'WorkLifeBalance': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.45, 0.25]),
    'YearsAtCompany': np.random.exponential(5, n_samples).clip(0, 30).astype(int),
    'YearsInCurrentRole': np.random.exponential(3, n_samples).clip(0, 15).astype(int),
    'YearsSinceLastPromotion': np.random.exponential(2, n_samples).clip(0, 10).astype(int),
    'YearsWithCurrManager': np.random.exponential(3, n_samples).clip(0, 15).astype(int),
}

df = pd.DataFrame(data)

# Generate realistic attrition based on patterns (target: ~16% attrition)
attrition_prob = (
    (df['OverTime'] == 'Yes') * 0.28 +           # Overtime biggest factor
    (df['JobSatisfaction'] <= 2) * 0.16 +         # Low satisfaction
    (df['WorkLifeBalance'] <= 2) * 0.14 +         # Poor WLB
    (df['YearsAtCompany'] < 2) * 0.12 +           # New hires
    (df['YearsSinceLastPromotion'] > 3) * 0.10 +  # Stagnation
    (df['MonthlyIncome'] < 4000) * 0.10 +         # Low pay
    (df['DistanceFromHome'] > 15) * 0.06 +        # Long commute
    (df['Age'] < 30) * 0.04                       # Young age
)

# Add random noise
attrition_prob = attrition_prob + np.random.uniform(-0.08, 0.08, n_samples)
attrition_prob = attrition_prob.clip(0, 0.9)

# Assign attrition to achieve ~16% rate
target_attrition_rate = 0.16
target_attrition_count = int(n_samples * target_attrition_rate)

# Sort by probability and assign attrition to highest risk employees
sorted_indices = np.argsort(attrition_prob.values)[::-1]
df['Attrition'] = 'No'
df.loc[sorted_indices[:target_attrition_count], 'Attrition'] = 'Yes'

# Verify rate
actual_rate = (df['Attrition'] == 'Yes').mean()
print(f"   ✓ Created dataset with {len(df)} employees")
print(f"   ✓ Attrition rate: {actual_rate*100:.1f}% (target: 16%)")
print(f"   ✓ Features: {len(df.columns)}")

# Step 2: Preprocessing
print("\n🔧 Step 2: Preprocessing data...")

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   ✓ Encoded: {col}")

# Step 3: Feature Selection
print("\n🎯 Step 3: Selecting features...")

feature_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

X = df[feature_columns]
y = df['Attrition']

print(f"   ✓ Features: {len(feature_columns)}")

# Step 4: Split Data
print("\n✂️ Step 4: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Training: {len(X_train)} samples")
print(f"   ✓ Testing: {len(X_test)} samples")

# Step 5: Scale Features
print("\n📐 Step 5: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Scaling complete")

# Step 6: Train XGBoost Model
print("\n🤖 Step 6: Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train, verbose=False)
print("   ✓ Training complete")

# Step 7: Evaluate Model
print("\n📊 Step 7: Evaluating model...")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print(" MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print("=" * 60)

# Step 8: Feature Importance
print("\n⭐ Step 8: Top 10 Most Important Features:")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i in range(10):
    row = feature_importance.iloc[i]
    bar = "█" * int(row['importance'] * 50)
    print(f"   {i+1:2d}. {row['feature']:25s} {bar:30s} {row['importance']:.4f}")

# Step 9: Save Model and Artifacts
print("\n💾 Step 9: Saving model and artifacts...")

os.makedirs('models', exist_ok=True)

with open('models/attrition_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ attrition_model.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ scaler.pkl")

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("   ✓ feature_columns.pkl")

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✓ label_encoders.pkl")

# Step 10: Save dataset for reference
os.makedirs('data', exist_ok=True)
df.to_csv('data/HR_Employee_Attrition.csv', index=False)
print("   ✓ dataset saved to data/HR_Employee_Attrition.csv")

# Step 11: Test Prediction
print("\n🧪 Step 10: Testing sample prediction...")

# Test with average employee
sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)[0]
prob = model.predict_proba(sample_scaled)[0][1]

print(f"   Sample employee prediction:")
print(f"   Risk Score: {prob*100:.1f}%")
print(f"   Will {'LEAVE' if pred == 1 else 'STAY'}")

# Test with high-risk employee
high_risk = X_test.iloc[0:1].copy()
high_risk['OverTime'] = 1  # Set overtime to Yes
high_risk['JobSatisfaction'] = 1  # Low satisfaction
high_risk_scaled = scaler.transform(high_risk)
high_risk_pred = model.predict(high_risk_scaled)[0]
high_risk_prob = model.predict_proba(high_risk_scaled)[0][1]

print(f"\n   High-risk scenario (Overtime + Low Satisfaction):")
print(f"   Risk Score: {high_risk_prob*100:.1f}%")
print(f"   Will {'LEAVE' if high_risk_pred == 1 else 'STAY'}")

print("\n" + "=" * 60)
print("✅ SETUP COMPLETE! Model is ready for production!")
print("=" * 60)
print("\n📁 Files created:")
print("   • models/attrition_model.pkl - Trained XGBoost model")
print("   • models/scaler.pkl - Feature scaler")
print("   • models/feature_columns.pkl - Feature names")
print("   • data/HR_Employee_Attrition.csv - Dataset")
print("\n🚀 To start the web application, run:")
print("   python app.py")
print("\n🌐 Then open: http://127.0.0.1:5000")
print("=" * 60)