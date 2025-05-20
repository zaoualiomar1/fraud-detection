print("Producer script started...")
from kafka import KafkaProducer
import json 
import pandas as pd 
import time 
import argparse 
import logging 
from kafka.errors import kafka_errors 

logger = logging.getLogger()
logging.basicConfig(filename= 'producer.log', level=logging.INFO)
# Reading the csv file 
logger.info('Reading the raw Data ')
try: 
    df = pd.read_csv(r"C:\Users\33787\Desktop\Omar\Projects\fraud-detection\data\raw-data\creditcard.csv")
    logger.info('Finished reading the file with sucess ')
     
except ValueError as error : 
    logger.info('Error reading the file', error )
    
    
print(f"Loaded {len(df)} transactions")

# Convert rows to dict for serilia
transactions = df.to_dict(orient='records')

# Kafka producer 

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks = 'all', 
    retries = 5
)

topic = 'transactions'

print(f"Starting to steam data {len(transactions)} transactions...")

parser = argparse.ArgumentParser()
parser.add_argument('--delay', type=float, default= 0.5, help= 'delay between messages ')
parser.add_argument('--limit', type = float, default= None, help= 'Define the maximum number of messages sent ')
args = parser.add_argument()

# Stream each transaction 
for txn in transactions : 
    try :
        producer.send( topic, txn)
    except kafka_errors as e : 
        logger.error(" kafka send failed for transaction {e}")
        
    print('produced', txn)
    time.sleep(0.5)

