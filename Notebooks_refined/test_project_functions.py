from project_functions import objective, app
from fastapi.testclient import TestClient
import optuna

def test_objective():
    accuracy = round(
        objective(
            optuna.trial.FixedTrial({"n_estimators": 800, "max_depth": 2, "min_child_weight": 10,
                                     "learning_rate": 0.03, "gamma": 0.01, "subsample": 0.7,
                                     "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 0.05})
                    ), 3)

    assert accuracy == 0.797


client = TestClient(app)
def test_predict():
    # Define test data
    test_data = {
       "tenure": 12,
       "MonthlyCharges": 20,
       "TotalCharges": 240,
       "TotalServices": 3,
       "gender_Male": 1,
       "InternetService_Fiber optic": 0,
       "InternetService_No": 1,
       "Contract_One year": 0,
       "Contract_Two year": 1,
       "PaymentMethod_Credit card (automatic)": 0,
       "PaymentMethod_Electronic check": 1,
       "PaymentMethod_Mailed check": 0,
       "SeniorCitizen": 0,
       "Partner": 1,
       "Dependents": 0,
       "PhoneService": 1,
       "MultipleLines": 0,
       "OnlineSecurity": 0,
       "OnlineBackup": 0,
       "DeviceProtection": 1,
       "TechSupport": 0,
       "StreamingTV": 0,
       "StreamingMovies": 0,
       "PaperlessBilling": 1,
    }
    
    # Send a POST request to the /predict endpoint
    response = client.post("/predict", json=test_data)
    
    # Check the status code
    assert response.status_code == 200
    
    # Check the returned prediction
    assert "prediction" in response.json()