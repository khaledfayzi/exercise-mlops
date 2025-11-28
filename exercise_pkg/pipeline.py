from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# aten vorbereiten
# diese funtkion lädet die daten und teilt sie in Train und Test.
def data_preparation(test_size=0.2):
    heart = fetch_ucirepo(id=45)
    x = heart.data.features.dropna()
    y = heart.data.targets.squeeze()[x.index]
    return train_test_split(x, y, test_size=test_size, random_state=42)


# Modell trainieren
# hier wird das model trainiert
def train(x_train, y_train):
    # Daten skalieren
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    # model erstellen
    model = LogisticRegression(max_iter=1000)
    # das model lernt muster aus den trainingsdaten
    model.fit(x_train_scaled, y_train)
    return model, scaler


# test, hier wird getestet, wie gut das gelernte Modell ist
def evaluate(model, scaler, x_test, y_test):
    # test daten skaliern
    x_test_scaled = scaler.transform(x_test)
    # model sagt krank/nicht krnak
    y_pred = model.predict(x_test_scaled)
    # genauigkeit berechnen , wie viele Vohersagen waren richtig
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy", acc)
    return acc


# Pipeline ausführen
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_preparation()
    model, scaler = train(x_train, y_train)
    evaluate(model, scaler, x_test, y_test)
