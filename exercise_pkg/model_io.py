import joblib
# model speichern
def save_model(model, scaler, model_path="model.joblib", scaler_path="scaler.joblib"):
    # objekt in datei schriben
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


# model wieder laden, damit man weiterarbeiten kann
def load_model(model_path="model.joblib", scaler_path="scaler.joblib"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
