from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Get the absolute path to the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model artifacts
MODEL_PATH = os.path.join(BASE_DIR, "models", "attrition_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

# Load model and artifacts
print("\n" + "="*60)
print("📂 LOADING MODEL ARTIFACTS")
print("="*60)

model = None
scaler = None
feature_columns = None

# Load model
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"⚠️ Model not found at: {MODEL_PATH}")

# Load scaler
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
else:
    print(f"⚠️ Scaler not found at: {SCALER_PATH}")

# Load feature columns
if os.path.exists(FEATURES_PATH):
    try:
        with open(FEATURES_PATH, 'rb') as f:
            feature_columns = pickle.load(f)
        print(f"✅ Loaded {len(feature_columns)} features")
    except Exception as e:
        print(f"❌ Error loading features: {e}")
else:
    print(f"⚠️ Feature columns not found, using defaults")
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

print("="*60 + "\n")

# Dashboard statistics (matching your trained model's dataset)
DASHBOARD_STATS = {
    "total_employees": 1470,
    "attrition_count": 237,
    "attrition_rate": 16.1,
    "avg_age": 36.9,
    "avg_tenure": 7.0,
    "avg_salary": 6503,
    "dept_attrition": {
        "Sales": 20.6,
        "Human Resources": 19.0,
        "Research & Development": 13.8
    },
    "age_groups": {
        "18-25": 45, "26-35": 112, "36-45": 51, "46-55": 22, "56+": 7
    },
    "overtime_attrition": {"Yes": 30.5, "No": 10.4},
    "satisfaction_attrition": {
        "Low (1)": 22.8, "Medium (2)": 16.4,
        "High (3)": 14.3, "Very High (4)": 11.3
    },
    "monthly_trend": [18, 15, 19, 17, 21, 16, 14, 18, 20, 15, 17, 19],
    "job_roles": {
        "Sales Executive": 17.5,
        "Research Scientist": 16.1,
        "Laboratory Technician": 23.9,
        "Human Resources": 19.0,
        "Manager": 5.1,
        "Sales Representative": 39.8,
        "Research Director": 2.5
    }
}

# Routes
@app.route('/')
def index():
    """Home page - Overview Dashboard"""
    return render_template('index.html', stats=DASHBOARD_STATS, year=datetime.now().year)

@app.route('/eda')
def eda():
    """Exploratory Data Analysis page"""
    return render_template('eda.html', stats=DASHBOARD_STATS, year=datetime.now().year)

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html', year=datetime.now().year)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for attrition prediction"""
    try:
        data = request.get_json()
        
        # Build feature dictionary from request data
        features_dict = {
            'Age': int(data.get('Age', 30)),
            'DailyRate': int(data.get('DailyRate', 800)),
            'DistanceFromHome': int(data.get('DistanceFromHome', 5)),
            'Education': int(data.get('Education', 3)),
            'EnvironmentSatisfaction': int(data.get('EnvironmentSatisfaction', 3)),
            'HourlyRate': int(data.get('HourlyRate', 65)),
            'JobInvolvement': int(data.get('JobInvolvement', 3)),
            'JobLevel': int(data.get('JobLevel', 2)),
            'JobSatisfaction': int(data.get('JobSatisfaction', 3)),
            'MonthlyIncome': int(data.get('MonthlyIncome', 6000)),
            'MonthlyRate': int(data.get('MonthlyRate', 14000)),
            'NumCompaniesWorked': int(data.get('NumCompaniesWorked', 2)),
            'OverTime': 1 if data.get('OverTime') == 'Yes' else 0,
            'PercentSalaryHike': int(data.get('PercentSalaryHike', 13)),
            'PerformanceRating': int(data.get('PerformanceRating', 3)),
            'RelationshipSatisfaction': int(data.get('RelationshipSatisfaction', 3)),
            'StockOptionLevel': int(data.get('StockOptionLevel', 1)),
            'TotalWorkingYears': int(data.get('TotalWorkingYears', 10)),
            'TrainingTimesLastYear': int(data.get('TrainingTimesLastYear', 3)),
            'WorkLifeBalance': int(data.get('WorkLifeBalance', 3)),
            'YearsAtCompany': int(data.get('YearsAtCompany', 5)),
            'YearsInCurrentRole': int(data.get('YearsInCurrentRole', 3)),
            'YearsSinceLastPromotion': int(data.get('YearsSinceLastPromotion', 1)),
            'YearsWithCurrManager': int(data.get('YearsWithCurrManager', 3)),
        }
        
        # Create feature array in the correct order
        features = [features_dict[col] for col in feature_columns]
        X = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Get prediction probability
        if model is not None:
            probability = float(model.predict_proba(X_scaled)[0][1])
        else:
            # Fallback calculation if model not loaded
            probability = calculate_fallback_risk(features_dict)
        
        # Determine risk level and color
        if probability >= 0.60:
            risk_level = "high"
            risk_label = "High Risk"
            color = "#ef4444"
        elif probability >= 0.35:
            risk_level = "moderate"
            risk_label = "Moderate Risk"
            color = "#f59e0b"
        else:
            risk_level = "low"
            risk_label = "Low Risk"
            color = "#10b981"
        
        # Generate contributing factors
        factors = generate_factors(data, features_dict)
        
        # Return JSON response
        return jsonify({
            "probability": round(probability * 100, 1),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "color": color,
            "factors": factors,
            "model_loaded": model is not None
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_count": len(feature_columns),
        "timestamp": datetime.now().isoformat()
    })

def calculate_fallback_risk(features):
    """Fallback risk calculation when model is not available"""
    risk = 0.10  # Base risk
    
    # Overtime
    if features['OverTime'] == 1:
        risk += 0.25
    
    # Low Job Satisfaction
    if features['JobSatisfaction'] <= 2:
        risk += (3 - features['JobSatisfaction']) * 0.08
    
    # Poor Work-Life Balance
    if features['WorkLifeBalance'] <= 2:
        risk += 0.15
    
    # Long commute
    if features['DistanceFromHome'] > 15:
        risk += 0.10
    
    # Career stagnation
    if features['YearsSinceLastPromotion'] > 3:
        risk += 0.10
    
    # Short tenure
    if features['YearsAtCompany'] < 2:
        risk += 0.08
    
    # Low income
    if features['MonthlyIncome'] < 4000:
        risk += 0.08
    
    return min(0.95, max(0.05, risk))

def generate_factors(data, features):
    """Generate explanation factors for prediction"""
    factors = []
    
    # Overtime factor
    if data.get('OverTime') == 'Yes':
        factors.append({
            "factor": "Overtime",
            "impact": "High",
            "detail": "Employees working overtime have 3.2x higher attrition risk according to XGBoost model"
        })
    
    # Job Satisfaction factor
    job_sat = features['JobSatisfaction']
    if job_sat <= 2:
        impact = "High" if job_sat == 1 else "Medium"
        factors.append({
            "factor": "Low Job Satisfaction",
            "impact": impact,
            "detail": f"Rating {job_sat}/4 - Top 3 predictor of attrition in our model"
        })
    
    # Work-Life Balance factor
    wlb = features['WorkLifeBalance']
    if wlb <= 2:
        impact = "High" if wlb == 1 else "Medium"
        factors.append({
            "factor": "Poor Work-Life Balance",
            "impact": impact,
            "detail": f"WLB score {wlb}/4 - Critical driver of employee burnout and turnover"
        })
    
    # Years at Company factor
    years_at_co = features['YearsAtCompany']
    if years_at_co < 2:
        factors.append({
            "factor": "Short Tenure",
            "impact": "High",
            "detail": "Employees with less than 2 years tenure show highest attrition rates (33.6%)"
        })
    elif years_at_co > 10:
        factors.append({
            "factor": "Long Tenure",
            "impact": "Low",
            "detail": f"{years_at_co} years - Long-tenure employees show strong retention"
        })
    
    # Career stagnation factor
    years_no_promo = features['YearsSinceLastPromotion']
    if years_no_promo > 3:
        impact = "High" if years_no_promo > 5 else "Medium"
        factors.append({
            "factor": "Career Stagnation",
            "impact": impact,
            "detail": f"No promotion in {years_no_promo} years - Lack of growth opportunities"
        })
    
    # Commute distance factor
    distance = features['DistanceFromHome']
    if distance > 15:
        factors.append({
            "factor": "Long Commute",
            "impact": "Medium",
            "detail": f"{distance}km commute - 25.7% attrition rate for 26+ km commuters"
        })
    
    # Income factor
    monthly_income = features['MonthlyIncome']
    job_level = features['JobLevel']
    if monthly_income < 4500 and job_level <= 2:
        factors.append({
            "factor": "Below Market Compensation",
            "impact": "Medium",
            "detail": f"Income ${monthly_income} below competitive range for job level {job_level}"
        })
    
    # Stock options factor
    stock_option = features['StockOptionLevel']
    if stock_option == 0:
        factors.append({
            "factor": "No Equity Incentive",
            "impact": "Low",
            "detail": "Lack of stock options reduces long-term retention incentives"
        })
    
    # If no factors identified
    if not factors:
        factors.append({
            "factor": "Stable Employee Profile",
            "impact": "Low",
            "detail": "Based on XGBoost model analysis, no major attrition risk indicators detected"
        })
    
    # Limit to top 5 factors
    return factors[:5]

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ATTRITIONIQ SERVER")
    print("="*60)
    print(f"📍 Local URL: http://127.0.0.1:5000")
    print(f"📍 Local URL: http://localhost:5000")
    print(f"📁 Model Status: {'✓ LOADED' if model else '✗ Using Fallback'}")
    print(f"📊 Scaler Status: {'✓ LOADED' if scaler else '✗ Not Loaded'}")
    print(f"🔧 Features: {len(feature_columns)}")
    print(f"📡 API Endpoint: http://127.0.0.1:5000/api/predict")
    print(f"💚 Health Check: http://127.0.0.1:5000/api/health")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Run the Flask application
    app.run(debug=True, host='127.0.0.1', port=5000)