import pandas as pd 
import numpy as np 
import mlflow 
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import classification_report, f1_score
import joblib 

## Push  
# Reading the data 
# df = pd.read_csv(r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\data\raw-data\creditcard.csv")
df = pd.read_csv('data/raw-data/creditcard.csv')

# Data preparation 
X = df.drop(columns=['Class'])
Y = df['Class']

scaler = StandardScaler()
# X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']]) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

models = [
    
    (
        "Logistic Regression", 
        {"C":1, 'solver' : 'liblinear'}, 
        LogisticRegression(), 
        (X_train, y_train), 
        (X_test, y_test)
    ),
    (   "Random Forest", 
        {"n_estimators" : 30,  'max_depth' : 3}, 
        RandomForestClassifier(),
        (X_train, y_train), 
        (X_test, y_test)
    ), 
    
    (   
        ('XGBClassifier'), 
        {"use_label_encoder": False, "eval_metric": 'logloss'}, 
        XGBClassifier(), 
        (X_train, y_train), 
        (X_test, y_test)
        
    ) ]

# Starting mlflow
mlflow.set_experiment('fraude-detection-modele-comparaison')
mlflow.set_tracking_uri("http://localhost:5000")
    
for i, element in enumerate(models): 
    model_name = element[0]
    params = element[1]
    model = element[2]
    report = report[i]

with mlflow.start_run():
    reports = []
    for model_name, params, model, train_set, test_set in models : 
        x_train = train_set[0]
        y_train = train_set[1]
        X_test = test_set[0]
        y_test = test_set[1]
    
        model.set_params(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)
    
    

    mlflow.log_param(params) 
    mlflow.log_metrics({
            'accuracy': report['accuracy'],
            'recall_class_1': report['1']['recall'],
            'recall_class_0': report['0']['recall'],
            'f1_score_macro': report['macro avg']['f1-score']
        })  
    
    if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
    else:
        mlflow.sklearn.log_model(model, "model")  
    

# with mlflow.start_run() : 
#     model = LogisticRegression(max_iter= 100)
#     model.fit(X_train, y_train)
    
#     y_pred = model.predict(X_test)
#     f1 = f1_score(y_test, y_pred)
    

    
    
    # mlflow.log_param("model_type", "LogisticRegression")
    # mlflow.log_metric("f1_score", f1)
    # mlflow.sklearn.log_model(model, "model")
    
    # print("F1 Score:", f1)
    # print(classification_report(y_test, y_pred))
    


# # Saving the model after prediction 
# print('Saving the model and the scaler -- -')
# joblib.dump(model, r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\model\model_trained\model.pkl")
# # joblib.dump(scaler, r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\model\model_trained\scaler.pkl")
# print('Model and scaler saved sucessufully!! -- -')




