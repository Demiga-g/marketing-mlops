import requests


details = {
    "ID":49979,"Year_Birth":1936.0,"Income":5242816.0,"Kidhome":4.0,
    "Teenhome":2,"Recency":2,"MntWines":800.0,"MntFruits":149,"MntMeatProducts":1094.0,"MntFishProducts":60,"MntSweetProducts":148.0,"MntGoldProds":19.0,"NumDealsPurchases":2,"NumWebPurchases":35,"NumCatalogPurchases":16,"NumStorePurchases":20,"NumWebVisitsMonth":4,"Complain":0,"AcceptedCmp3":0,"AcceptedCmp4":0,"AcceptedCmp5":0,"AcceptedCmp1":0,"AcceptedCmp2":0,"Z_CostContact":1,"Z_Revenue":1,"Education":"2n Cycle","Marital_Status":"Married","Dt_Customer":"2012-03-05"
}

url = 'http://localhost:9696/03-predict-model'
response = requests.post(url, json=details, timeout=30)
print(response.json())