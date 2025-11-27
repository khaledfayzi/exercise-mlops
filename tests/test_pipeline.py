import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipeline import data_preparation, train, evaluate


#testet, gibt es überhupt Daten? Ja -> test bestanden
def test_data_preparation():
    X_train, X_test, y_train, y_test = data_preparation()
    assert len(X_train) > 0
    assert len(X_test) > 0

#Ist das Modell testbar und gibt es eine normale Accuracy zurück?
def test_training_and_evaluation():
    X_train,X_test,y_train,y_test=data_preparation()
    model,scaler=train(X_train,y_train)
    acc=evaluate(model,scaler,X_test,y_test)

    assert 0<=acc <=1
