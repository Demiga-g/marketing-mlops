import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



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
    return 1.0
    
    
def lambda_handler(event, context):
    details = json.dumps(event, indent=4)
    cust_id = event['ID']
    
    features = scrub_data(details)
    prediction = predict(features)
    
    # print(json.dumps(event))
    return {
        'cust_id': cust_id,
        'response': prediction
    }


{
    "ID": 49979,
    "Year_Birth": 1936,
    "Income": 5242816,
    "Kidhome": 4,
    "Teenhome": 2,
    "Recency": 2,
    "MntWines": 800,
    "MntFruits": 149,
    "MntMeatProducts": 1094,
    "MntFishProducts": 300,
    "MntSweetProducts": 148,
    "MntGoldProds": 19,
    "NumDealsPurchases": 2,
    "NumWebPurchases": 5,
    "NumCatalogPurchases": 16,
    "NumStorePurchases": 20,
    "NumWebVisitsMonth": 4,
    "Complain": 0,
    "AcceptedCmp3": 0,
    "AcceptedCmp4": 0,
    "AcceptedCmp5": 0,
    "AcceptedCmp1": 0,
    "AcceptedCmp2": 0,
    "Z_CostContact": 1,
    "Z_Revenue": 1,
    "Education": "2n Cycle",
    "Marital_Status": "Married",
    "Dt_Customer": "2012-03-05"
}










def scrub_data(details):
    
    df = pd.DataFrame([details])
        
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
    return 1.0
    
    
def lambda_handler(event, context):
    details = json.dumps(event)
    
    features = scrub_data(details)
    prediction = predict(features)
    
    # print(json.dumps(event))
    return {
        'response': prediction
    }









def scrub_data(details):
    
    df = pd.DataFrame([details])
        
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
    red_ftrs_1 = ["Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue",'Age', 'Response', 'ID']
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
    return 1.0
    
    
def lambda_handler(event, context):
    details = event['details']
    cust_id = event['cust_id']
    
    features = scrub_data(details)
    prediction = predict(features)
    
    # print(json.dumps(event))
    return {
        'response': prediction,
        'cust_id': cust_id
    }


{
  "details": {
    "ID": 49979,
    "Year_Birth": 1936,
    "Income": 5242816,
    "Kidhome": 4,
    "Teenhome": 2,
    "Recency": 2,
    "MntWines": 800,
    "MntFruits": 149,
    "MntMeatProducts": 1094,
    "MntFishProducts": 300,
    "MntSweetProducts": 148,
    "MntGoldProds": 19,
    "NumDealsPurchases": 2,
    "NumWebPurchases": 5,
    "NumCatalogPurchases": 16,
    "NumStorePurchases": 20,
    "NumWebVisitsMonth": 4,
    "Complain": 0,
    "AcceptedCmp3": 0,
    "AcceptedCmp4": 0,
    "AcceptedCmp5": 0,
    "AcceptedCmp1": 0,
    "AcceptedCmp2": 0,
    "Z_CostContact": 1,
    "Z_Revenue": 1,
    "Education": "2n Cycle",
    "Marital_Status": "Married",
    "Dt_Customer": "2012-03-05"
  },
  "cust_id": 123
}