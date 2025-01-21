import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle

def train_model(data_file, model_file):
    # Load data
    df = pd.read_csv(data_file)
    X = df[["Temperature", "Run_Time"]]
    y = df["Downtime_Flag"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Save model
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    return metrics

def predict_downtime(model_file, input_data):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([[input_data["Temperature"], input_data["Run_Time"]]])
    probability = model.predict_proba([[input_data["Temperature"], input_data["Run_Time"]]])[0].max()
    return {
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": round(probability, 2)
    }
