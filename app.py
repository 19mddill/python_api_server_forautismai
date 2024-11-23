from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
import joblib

app = Flask(__name__)

# Dataset path
data_path = "./data/data.csv"

# Initialize the model and encoders
stacked_model = None
label_encoders = {}
selected_features = None

@app.route('/train', methods=['POST'])
def train_model():
    global stacked_model, label_encoders, selected_features

    # Load the dataset
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500

    # Drop irrelevant columns
    df = data.drop(columns=["CASE_NO_PATIENT'S"]).copy()

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = df[column].fillna("Unknown")
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

    # Separate features and target
    X = df.drop(columns=["ASD_traits"])
    y = df["ASD_traits"]

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # MinMax Scaling and feature selection
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X_resampled)

    k = 25
    chi2_selector = SelectKBest(chi2, k=k)
    X_selected = chi2_selector.fit_transform(X_minmax, y_resampled)
    selected_features = chi2_selector

    # Model hyperparameters
    param_grid_rf = {
        'n_estimators': [400],
        'max_depth': [20],
        'min_samples_split': [2]
    }

    param_grid_xgb = {
        'n_estimators': [300],
        'learning_rate': [0.01],
        'max_depth': [7]
    }

    # Train individual models
    rf_tuned = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, scoring='accuracy', cv=3).fit(X_selected, y_resampled).best_estimator_
    xgb_tuned = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_grid_xgb, scoring='accuracy', cv=3).fit(X_selected, y_resampled).best_estimator_
    svc = SVC(C=3.0, kernel='linear', probability=True, random_state=42)
    ada = AdaBoostClassifier(n_estimators=200, algorithm='SAMME', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, learning_rate_init=0.001, random_state=42)

    # Stacking Classifier
    stacked_model = StackingClassifier(
        estimators=[
            ('Random Forest', rf_tuned),
            ('XGBoost', xgb_tuned),
            ('SVM', svc),
            ('AdaBoost', ada)
        ],
        final_estimator=MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, learning_rate_init=0.001, random_state=42),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )

    # Train the stacked model
    stacked_model.fit(X_selected, y_resampled)

    # Save the trained model and encoders
    joblib.dump(stacked_model, "model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    joblib.dump(selected_features, "selected_features.pkl")

    return jsonify({"message": "Model trained successfully"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    global stacked_model, label_encoders, selected_features

    # Load model and encoders if not already loaded
    if stacked_model is None:
        try:
            stacked_model = joblib.load("model.pkl")
            label_encoders = joblib.load("label_encoders.pkl")
            selected_features = joblib.load("selected_features.pkl")
        except Exception as e:
            return jsonify({"error": f"Failed to load the model: {str(e)}"}), 500

    # Get input data
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    # Create a dataframe for input
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    for column, le in label_encoders.items():
        if column in input_df:
            input_df[column] = input_df[column].fillna("Unknown")
            input_df[column] = le.transform(input_df[column])

    # Handle missing numerical values
    input_df = input_df.fillna(0)

    # Select features
    X_input = input_df.drop(columns=["ASD_traits"], errors='ignore')
    X_input_scaled = MinMaxScaler().fit_transform(X_input)
    X_input_selected = selected_features.transform(X_input_scaled)

    # Predict using the model
    prediction = stacked_model.predict(X_input_selected)

    return jsonify({"prediction": int(prediction[0])}), 200


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))  # Use the Render-provided PORT or default to 5000
    app.run(host="0.0.0.0", port=port)

