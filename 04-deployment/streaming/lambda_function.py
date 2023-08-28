import os
import json
import boto3
import mlflow
import base64
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

os.environ["AWS_PROFILE"] = "demiga-g"
RUN_ID = os.getenv('RUN_ID')
logged_model = f's3://ifood-deploy-01/1/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)

kinesis_client = boto3.client('kinesis')
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'response_prediction')

TEST_RUN = os.getenv('DRY_RUN', 'False') == 'True'

def scrub_data(details):
    
    df = pd.read_json(details, orient='index').transpose()
        
    # Convert 'Kidhome' and 'Teenhome' to categorical
    # but first fillna with the most frequent value
    dependants = ['Kidhome', 'Teenhome']
    df[dependants] = df[dependants].fillna(df[dependants].mode().iloc[0])
    df[dependants] = df[dependants].applymap(lambda x: 1 if x > 0 else 0)
    
    # Conversions into 'datetime' data type
    # but first fillna in both variables
    df['Year_Birth'] = df['Year_Birth'].fillna(int(df['Year_Birth'].median()))
    df['Year_Birth'] = pd.to_datetime(df['Year_Birth'], format='%Y')
    
    df['Dt_Customer'] = df['Dt_Customer'].fillna(df['Dt_Customer'].mode().iloc[0])
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    
    # Calculate age
    # assuming analysis was conducted in 2014 
    now = 2014
    df['Age'] = now - df['Year_Birth'].dt.year
    
    # Define the bin edges
    bins = [18, 28, 38, 48, 58, 65, np.inf]
    # Define the labels for each age group
    labels = ['18-27', '28-37', '38-47', '48-57', '58-65', '65+']
    # Create age group feature
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # Calculate the number of days since customer enrolled
    end_fiscal = datetime(2014, 6, 30)
    df['Onboard_Days'] = (end_fiscal - df['Dt_Customer']).dt.days
    
    # Droping redundant features
    red_ftrs_1 = ["Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue",'Age', 'ID']
    df = df.drop(red_ftrs_1, axis=1)
    
    # List of categorical and numeric features
    categ_ftrs_1 = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Age_Group']

    num_ftrs_1 = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Onboard_Days']
    
    # handle missing values and scale numeric data
    ct = ColumnTransformer([
        ('num_trans', SimpleImputer(strategy='median'), num_ftrs_1),
        ('cat_trans', SimpleImputer(strategy='most_frequent'), categ_ftrs_1)
    ])
        
    df = pd.DataFrame(ct.fit_transform(df), 
                      columns=num_ftrs_1+categ_ftrs_1)
    
    # Ensure that the final df features are in the right data types
    df[categ_ftrs_1] = df[categ_ftrs_1].astype('str')
    df[num_ftrs_1] = df[num_ftrs_1].astype('float')
     
    return df.to_dict(orient='records')

def predict(features):
    pred = model.predict(features)
    return float(pred[0])
    
def lambda_handler(event, context):

    # print(json.dumps(event))
    
    predictions = []
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        response_event = json.loads(decoded_data)
        # print(response_event)
        details = json.dumps(response_event, indent=4)
        cust_id = response_event['ID']
    
        features = scrub_data(details)
        prediction = predict(features)
        
        prediction_event = {
            'model': 'response_prediction_model',
            'version': '123',
            'prediction': {
                'cust_id': cust_id,
                'response': prediction
            }
                
        }
        
        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(cust_id)
            )   
           
        
        predictions.append(prediction_event)

    
    return {
        'predictions': predictions
    }
