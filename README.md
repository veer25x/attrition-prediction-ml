# attrition-prediction-ml
ML-powered HR analytics platform predicting employee attrition risk using XGBoost. Features interactive dashboard, real-time predictions, and SHAP-like explanations. 87%+ accuracy.

# AttritionIQ

A machine learning web app that predicts employee attrition risk using XGBoost. Built with Flask, it takes in HR data and returns a risk score along with the key factors driving that prediction.

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92.86% |
| ROC-AUC | 97.36% |
| Precision | 82.50% |
| Recall | 70.21% |
| F1 Score | 75.86% |

Trained on the IBM HR Analytics dataset (1,470 employees, 16.1% attrition rate).

## Tech Stack

- **ML:** XGBoost, Scikit-learn
- **Backend:** Flask
- **Data:** Pandas, NumPy
- **Frontend:** Chart.js

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/attrition-prediction-ml.git
cd attrition-prediction-ml
pip install -r requirements.txt
python setup_and_train.py
python app.py
```

App runs at `http://localhost:5000`

## API

```bash
POST /api/predict
Content-Type: application/json

{
  "Age": 32,
  "MonthlyIncome": 6500,
  "YearsAtCompany": 4,
  "JobSatisfaction": 3,
  "WorkLifeBalance": 3,
  "OverTime": "Yes"
}
```

## Dataset

IBM HR Analytics Employee Attrition Dataset — 1,470 records, 24 features.

## License

MIT
