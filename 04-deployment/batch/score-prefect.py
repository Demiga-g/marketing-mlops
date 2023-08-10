#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import os
import sys
import mlflow

import numpy as np
import pandas as pd

from datetime import datetime

from prefect import task, flow, get_run_logger

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer


os.environ["AWS_PROFILE"] = "demiga-g"

def read_clean_data(filename: str) -> pd.DataFrame:
    
    """Read in & clean the data"""
    df = pd.read_csv(filename)
    
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
    red_ftrs_1 = ["Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue",'Age']
    df = df.drop(red_ftrs_1, axis=1)
    
    return df


def miss_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values then normalize data and return a dictionary
    """
    
    # List of categorical and numeric features
    categ_ftrs_1 = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Age_Group']

    num_ftrs_1 = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Onboard_Days']
    
    # Drop the target and the ID variables
    df = df.drop(['Response', 'ID'], axis=1)
        
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('normalize', PowerTransformer(method='yeo-johnson')),
    ])
    
    ct = ColumnTransformer([
        ('num_trans', num_transformer, num_ftrs_1),
        ('cat_trans', SimpleImputer(strategy='most_frequent'), categ_ftrs_1)
    ])
        
    df = pd.DataFrame(ct.fit_transform(df), 
                      columns=num_ftrs_1+categ_ftrs_1)
        
    # Ensure that the final df features are in the right data types
    df[categ_ftrs_1] = df[categ_ftrs_1].astype('str')
    df[num_ftrs_1] = df[num_ftrs_1].astype('float')
    
     
    # Return a dictionary    
    return df.to_dict(orient='records')


def load_model(run_id):
    # BASE_LOCATION = 'home/mgubuntu/projects/marketing-mlops/04-deployment/web-service-mlflow/artifacts_local/1'
    BASE_LOCATION = 's3://ifood-deploy-01/1'
    logged_model = f'{BASE_LOCATION}/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result['CustomerID'] = df['ID']
    df_result['ActualResponse'] = df['Response']
    df_result['PredictedResponse'] = y_pred
    df_result['PredictionStatus'] = (df_result['ActualResponse'] == df_result['PredictedResponse']).map({True: 'Correct', False: 'Incorrect'})
    df_result['ModelVersion'] = run_id
    
    df_result.to_csv(output_file, index=False)

@task   
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()
    logger.info(f'reading the data from {input_file}...')
    df = read_clean_data(input_file)
    dicts = miss_norm(df)
    
    logger.info(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)
    
    logger.info('applying the model...')
    y_pred = model.predict(dicts)
    
    logger.info(f'saving the results to {output_file}')
    save_results(df, y_pred, run_id, output_file)
    return output_file


def get_paths(data_input):
    input_file = f'https://raw.githubusercontent.com/Demiga-g/marketing-mlops/main/01-initial-model/data/{data_input}.csv'
    output_file = f's3://ifood-data/{data_input}.csv'
    
    return input_file, output_file
    
    
@flow
def response_prediction(data_input: str, run_id: str):
    if data_input == 'training_data':
        data_input = 'validation_data'
    elif data_input == 'validation_data':
        data_input = 'training_data'

    input_file, output_file = get_paths(data_input)
    
    apply_model(
        input_file=input_file, 
        run_id=run_id, 
        output_file=output_file
    )
   

def run():
    data_input = sys.argv[1] #'validation_data'
    RUN_ID = sys.argv[2] #'47e954b2eafb4e72af30c898d0d32dcc'

    response_prediction(data_input=data_input, run_id=RUN_ID)

if __name__ == '__main__':
    run()