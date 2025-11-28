import os
import sys

# 🔥 Projekt-Root zum Python-Pfad hinzufügen
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_io import save_model, load_model
from pipeline import data_preparation, train

def test_save_and_load_model():
    x_train, x_test, y_train, y_test = data_preparation()
    model, scaler = train(x_train, y_train)
    save_model(model, scaler)
    assert os.path.exists("model.joblib")
    assert os.path.exists("scaler.joblib")
    model2, scaler2 = load_model()
    assert model2 is not None
    assert scaler2 is not None
    # Aufräumen: ✅ richtige Dateien löschen
    os.remove("model.joblib")
    os.remove("scaler.joblib")
