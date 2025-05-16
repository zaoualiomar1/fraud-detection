print("Producer script started...")
from kafka import KafkaProducer
import json 
import sys 
import pandas as pd 
import time 

# Reading the csv file 

df = pd.read_csv(r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\data\raw-data\creditcard.csv")
print(f"Loaded {len(df)} transactions")

# Convert rows to dict for serilia
transactions = df.to_dict(orient='records')

# Kafka producer 

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


topic = 'transactions'

print(f"Starting to steam data {len(transactions)} transactions...")

# Stream each transaction 
for txn in transactions : 
    producer.send( topic, txn)
    print('produced', txn)
    time.sleep(0.5)

