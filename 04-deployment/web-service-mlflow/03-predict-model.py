import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
RUN_ID = "16fb592a742e430a92f6e6a5eee58c45"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

# Create the necessary variables
dependants = ['Kidhome', 'Teenhome']

# assuming analysis was conducted in 2014 
now = 2014

# Define the bin edges
bins = [18, 28, 38, 48, 58, 65, np.inf]

# Define the labels for each age group
labels = ['18-27', '28-37', '38-47', '48-57', '58-65', '65+']

# End of financial year
end_fiscal = datetime(2014, 6, 30)

# Redundant features
red_ftrs_1 = ["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue","Age"]

# List of categorical and numeric features
categ_ftrs_1 = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Age_Group']

num_ftrs_1 = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Onboard_Days']


# Function to do data cleaning and feature preprocessing
def scrub_data(details):
    
    df = pd.DataFrame([details])
        
    # Convert 'Kidhome' and 'Teenhome' to categorical
    # but first fillna with the most frequent value
    df[dependants] = df[dependants].fillna(df[dependants].mode().iloc[0])
    df[dependants] = df[dependants].applymap(lambda x: 1 if x > 0 else 0)
    
    # Conversions into 'datetime' data type
    # but first fillna in both variables
    df['Year_Birth'] = df['Year_Birth'].fillna(int(df['Year_Birth'].median()))
    df['Year_Birth'] = pd.to_datetime(df['Year_Birth'], format='%Y')
    
    df['Dt_Customer'] = df['Dt_Customer'].fillna(df['Dt_Customer'].mode().iloc[0])
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    
    # Calculate age
    df['Age'] = now - df['Year_Birth'].dt.year
    
    # Create age group feature
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # Calculate the number of days since customer enrolled
    df['Onboard_Days'] = (end_fiscal - df['Dt_Customer']).dt.days
    
    # Droping redundant features
    df = df.drop(red_ftrs_1, axis=1)
    
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
    preds = model.predict(features)
    return float(preds[0])


app = Flask('customer-response')

@app.route('/03-predict-model', methods=['POST'])
def predict_endpoint():
    details = request.get_json()
    
    features = scrub_data(details)
    pred = predict(features)
    
    result = {
        'response': pred
    }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)