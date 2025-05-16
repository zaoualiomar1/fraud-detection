import pandas as pd 
import numpy as np 
import mlflow 
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib 


# Reading the data 
df = pd.read_csv(r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\data\raw-data\creditcard.csv")

# Data preparation 
X = df.drop(columns=['Class'])
Y = df['Class']

scaler = StandardScaler()
# X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']]) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)


# Starting mlflow 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")

with mlflow.start_run() : 
    model = LogisticRegression(max_iter= 100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")
    
    print("F1 Score:", f1)
    print(classification_report(y_test, y_pred))
    


# Saving the model after prediction 
print('Saving the model and the scaler -- -')
joblib.dump(model, r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\model\model_trained\model.pkl")
# joblib.dump(scaler, r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\model\model_trained\scaler.pkl")
print('Model and scaler saved sucessufully!! -- -')




