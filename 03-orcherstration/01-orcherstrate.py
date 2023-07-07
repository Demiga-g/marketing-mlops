import scipy
import mlflow
import pickle
import pathlib
import sklearn
import numpy as np
import pandas as pd
from prefect import flow, task

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score,
                             recall_score,
                             roc_auc_score,
                             accuracy_score, 
                             precision_score)



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
    red_ftrs_1 = ["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue",'Age']
    df = df.drop(red_ftrs_1, axis=1)
    
    return df
    
    
def miss_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values then normalize data"""
    
    # List of categorical and numeric features
    categ_ftrs_1 = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Age_Group']

    num_ftrs_1 = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Onboard_Days']
    
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
    
    return df



def vectorize(train_data: pd.DataFrame, val_data: pd.DataFrame) -> tuple([
    scipy.sparse._csr.csr_matrix, 
    scipy.sparse._csr.csr_matrix, 
    np.ndarray, 
    np.ndarray, 
    sklearn.feature_extraction.DictVectorizer]):
    
    """Vectorize the training and validation data"""
    
    # From DataFrame to dictionary
    train_dicts= train_data.to_dict(orient='records')
    val_dicts = val_data.to_dict(orient='records')
    
    # Use dictionary vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    # Target variables
    y_train = train_data['Response'].values
    y_val = val_data['Response'].values
    
    return X_train, X_val, y_train, y_val, dv


def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix, 
    X_val: scipy._csr.csr_matrix, 
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    dv: sklearn.feature_extraction.DictVectorizer
    ) -> None:
    """Train a model with best hyperparameters"""
    
    # Train the best model and log it in mlflow
    with mlflow.start_run():
        params = {
            'min_samples_leaf': 8, 
            'min_samples_split': 14, 
            'n_estimators': 90
        }
        
        # Log the parameters
        mlflow.log_params(params)

        gbc = GradientBoostingClassifier(**params, random_state=42)
        gbc.fit(X_train, y_train)
        y_pred = gbc.predict(X_val)
        
        # Calculate the evaluation metrics
        metrics = {
            'f1': f1_score(y_val, y_pred.round()), 
            'precision': precision_score(y_val, y_pred.round(), zero_division=0),
            'recall': recall_score(y_val, y_pred.round()),
            'pr_auc': roc_auc_score(y_val, y_pred.round()),
            'accuracy': accuracy_score(y_val, y_pred.round())
        }
        
        # Log the evaluation metrics
        mlflow.log_metrics(metrics)
        
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
    
        # Log the preprocessor
        mlflow.log_artifact('models/preprocessor.b', artifact_path='preprocessor')
        
        # Log the model
        mlflow.sklearn.log_model(gbc, artifact_path='model')
        
        return None
    
    
