# Steps:
# 1) Data Collection: Obtain the dataset from TechElectro Inc., containing customer information,
# purchase history, demographics, and preferences.

import pandas as pd
import random
#I am making the dataset myself, starting with values given in question & then adding random values
size=503
customerINFO = {
    'CustomerID': [1001, 1002, 1003, 1004, 1005,1006,1007,1008,1009,1010]
    +[i for i in range (1008,1010+size-7)], #Adding duplicates on purpose
    
    'Age': [33, 28, 42, 51, 37]
    +[20,-40] #Adding inconsistent value on purpose
    +[random.randint(10,100) for i in range(size-4)]+[None for i in range (2)],
    
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
    +[random.choice(['Male', 'Female']) for i in range (size-1)]+[None for i in range(1)],
    
    'MaritalStatus': ['Married', 'Single', 'Single', 'Married', 'Divorced']
    +['status'] #Adding inconsistent value on purpose
    +[random.choice(['Married', 'Single', 'Divorced']) for i  in range (size-4)]
    +[None for j in range (3)], #Adding null values on purpose
    
    'AnnualIncome (USD)': [65000, 45000, 55000, 80000, 58000]
    +[random.randint(30000,500000) for i in range (size-2)]+[None for j in range(2)],
    
    'TotalPurchases': [18, 15, 20, 12, 10]
    +[random.randint(0,100000)for i in range(size-2)]+[None for j in range (2)],
    
    'PreferredCategory': ['Electronics', 'Appliances', 'Electronics', 'Electronics', 'Appliances']
    +[random.choice(['Electronics',
                     'Appliances']) for i in range(size)] #Assuming possible categories
}

# Creating a pandas DataFrame
df = pd.DataFrame(customerINFO)
df.head(8)
# Saving the DataFrame to a CSV file
df.to_csv('TechElectro_Customer_Data.csv', index=False)
