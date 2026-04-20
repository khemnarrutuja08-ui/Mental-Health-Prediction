# ===============================
# 1. Import Libraries
# ===============================
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os

# ===============================
# 2. Initialize Flask App
# ===============================
app = Flask(__name__)
app.secret_key = "mental_health_secret_key"

# ===============================
# 3. Load ML Model Files
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
label_encoders = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))
target_encoder = pickle.load(open(os.path.join(BASE_DIR, "target_encoder.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# ===============================
# 4. Home Page
# ===============================
@app.route('/')
def home():
    return render_template("index.html")

# ===============================
# 5. Login Page
# ===============================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Password match check
        if password != confirm_password:
            return "Passwords do not match. Please try again."

        print("Login successful for:", username)

        return redirect(url_for('evaluation'))

    return render_template("login.html")

# ===============================
# 6. Register Page
# ===============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Password match check
        if password != confirm_password:
            return "Passwords do not match. Please try again."

        print("User registered:", username)

        return redirect(url_for('login'))

    return render_template("register.html")

# ===============================
# 7. Evaluation Page
# ===============================
@app.route('/evaluation')
def evaluation():
    return render_template("evaluation.html")

# ===============================
# 8. Prediction Route
# ===============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numerical Inputs
        age = float(request.form['age'])
        stress_level = float(request.form['stress_level'])
        sleep_hours = float(request.form['sleep_hours'])
        physical_activity_days = float(request.form['physical_activity_days'])
        depression_score = float(request.form['depression_score'])
        anxiety_score = float(request.form['anxiety_score'])
        social_support_score = float(request.form['social_support_score'])
        productivity_score = float(request.form['productivity_score'])

        # Categorical Encoding
        gender = label_encoders['gender'].transform([request.form['gender']])[0]
        employment_status = label_encoders['employment_status'].transform([request.form['employment_status']])[0]
        work_environment = label_encoders['work_environment'].transform([request.form['work_environment']])[0]
        mental_health_history = label_encoders['mental_health_history'].transform([request.form['mental_health_history']])[0]
        seeks_treatment = label_encoders['seeks_treatment'].transform([request.form['seeks_treatment']])[0]

        # Arrange Input
        input_data = np.array([[
            age,
            gender,
            employment_status,
            work_environment,
            mental_health_history,
            seeks_treatment,
            stress_level,
            sleep_hours,
            physical_activity_days,
            depression_score,
            anxiety_score,
            social_support_score,
            productivity_score
        ]])

        # Scale Input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        result = target_encoder.inverse_transform(prediction)[0]

        # Risk Percentage for Graph
        if result.lower() == "high":
            risk_value = 85
        elif result.lower() == "medium":
            risk_value = 50
        else:
            risk_value = 20

        return render_template(
            "dashboard.html",
            prediction_text="Mental Health Risk: " + result,
            risk_percent=risk_value
        )

    except Exception as e:
        return "Error occurred: " + str(e)

# ===============================
# 9. Run Application
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
