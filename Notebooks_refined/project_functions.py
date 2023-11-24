#IMPORTS
import optuna
import pickle
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score #I added this last one
from sklearn.metrics import accuracy_score, classification_report

from fastapi import FastAPI
from pydantic import BaseModel, Field

df = pd.read_csv('../data/Telco_ML_ready.csv')
X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df['Churn'], test_size=0.3, random_state=1)

#FUNCTIONS
def objective(trial):
   xgb_model = XGBClassifier(
       n_estimators=trial.suggest_int("n_estimators", 100, 1200),
       max_depth=trial.suggest_int("max_depth", 1, 10),
       min_child_weight=trial.suggest_float("min_child_weight", 0, 20),
       learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
       gamma=trial.suggest_float("gamma", 1e-5, 1, log=True),
       subsample=trial.suggest_float("subsample", 0.05, 1.0),
       colsample_bytree=trial.suggest_float("colsample_bytree", 0.05, 1.0),
       reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 1e-1, log=True),
       reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 1e-1, log=True),
       random_state=99
   )

   score = cross_val_score(xgb_model, X_train, y_train, cv=3)
   accuracy = score.mean()
    
   return accuracy

class Data(BaseModel):
    tenure	:	int
    MonthlyCharges	:	int
    TotalCharges	:	float
    TotalServices	:	float
    gender_Male	:	int
    InternetService_Fiber_optic	:	int = Field(alias='InternetService_Fiber optic')
    InternetService_No	:	int
    Contract_One_year	:	int = Field(alias='Contract_One year')
    Contract_Two_year	:	int = Field(alias='Contract_Two year')
    PaymentMethod_Credit_card_automatic	:	int = Field(alias='PaymentMethod_Credit card (automatic)')
    PaymentMethod_Electronic_check	:	int = Field(alias='PaymentMethod_Electronic check')
    PaymentMethod_Mailed_check	:	int = Field(alias='PaymentMethod_Mailed check')
    SeniorCitizen	:	int
    Partner	:	int
    Dependents	:	int
    PhoneService	:	int
    MultipleLines	:	int
    OnlineSecurity	:	int
    OnlineBackup	:	int
    DeviceProtection	:	int
    TechSupport	:	int
    StreamingTV	:	int
    StreamingMovies	:	int
    PaperlessBilling	:	int

   # Add more features as needed

with open('xgb_model.pkl', 'rb') as file:
   model = pickle.load(file)

app = FastAPI()

@app.post('/predict')
def predict(data: Data):
   # Prepare data for prediction
   data_to_predict = pd.DataFrame(data.dict(), index=[0]).values
    
   # Predict the class
   prediction = model.predict(data_to_predict)
   prediction = {'prediction': str(prediction[0])}

   # Return the result
   return prediction