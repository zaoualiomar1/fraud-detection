import os 
import kaggle 


# Set Kaggle credentials from env
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')


kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path='data/raw-data', unzip=True)
print("✅ Downloaded creditcard.csv from Kaggle")