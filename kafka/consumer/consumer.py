import numpy as np
import joblib
import json
from kafka import KafkaConsumer
import mlflow 
import os 

# model = joblib.load(r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\model\model_trained\model.pkl")
scaler = joblib.load(r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\model\model_trained\scaler.pkl")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
model_name = os.getenv("MODEL_NAME", "fraud-detector-model")
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")


# model = mlflow.pyfunc.load_model("models:/fraud-detector-model/Production")

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    group_id='fraud-detector',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

feature_columns = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
    'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
    'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

print("🚀 Real-time fraud detection started...\n")

for message in consumer:
    
    txn = message.value

    print("🚀 Starting the try  ...\n")
    try:
        # Safely extract features
        features_raw = [txn[col] for col in feature_columns]
        features = np.array(features_raw).reshape(1, -1)

        # Scale only Time and Amount
        print("🚀 Feature cleaning ...\n")
        time_amount_indices = [0, -1]  # 'Time' and 'Amount'
        time_amount = features[:, time_amount_indices]  # shape (1, 2)
        time_amount_scaled = scaler.transform(time_amount)

        # Inject scaled values back
        features_scaled = features.copy()
        features_scaled[0, time_amount_indices[0]] = time_amount_scaled[0, 0]  # Time
        features_scaled[0, time_amount_indices[1]] = time_amount_scaled[0, 1]  # Amount

        # Predict
        print("🚀 Launching prediction on arriving data ...\n")
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0][1]

        if prediction == 1:
            print(f"⚠️ FRAUD DETECTED [Confidence: {confidence:.2f}] → {txn}")
        else:
            print(f"✅ Legit transaction [Confidence: {confidence:.2f}]")

    except Exception as e:
        print("❌ Error processing transaction:", e)
