import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv("mental_health_dataset.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("Dataset Loaded Successfully!\n")


# ==============================
# 3. Encode Categorical Columns
# ==============================

categorical_cols = [
    'gender',
    'employment_status',
    'work_environment',
    'mental_health_history',
    'seeks_treatment'
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("Categorical Columns Encoded Successfully!\n")


# ==============================
# 4. Encode Target Column
# ==============================

target_encoder = LabelEncoder()
data['mental_health_risk'] = target_encoder.fit_transform(data['mental_health_risk'])

print("Target Column Encoded Successfully!\n")


# ==============================
# 5. Define Features & Target
# ==============================

feature_columns = [
    'age',
    'gender',
    'employment_status',
    'work_environment',
    'mental_health_history',
    'seeks_treatment',
    'stress_level',
    'sleep_hours',
    'physical_activity_days',
    'depression_score',
    'anxiety_score',
    'social_support_score',
    'productivity_score'
]

X = data[feature_columns]
y = data['mental_health_risk']


# ==============================
# 6. Scale Features
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature Scaling Completed!\n")


# ==============================
# 7. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Train-Test Split Done!\n")


# ==============================
# 8. Train Model
# ==============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Training Completed!\n")


# ==============================
# 9. Model Evaluation
# ==============================

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm, "\n")

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))


# ==============================
# 10. Cross Validation
# ==============================

cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("Cross Validation Accuracy:", round(cv_scores.mean() * 100, 2), "%\n")


# ==============================
# 11. Save All Files
# ==============================

pickle.dump(label_encoders, open("label_encoder.pkl", "wb"))
pickle.dump(target_encoder, open("target_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print(" All files saved successfully!")
