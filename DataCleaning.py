import pandas as pd
data=pd.read_csv('TechElectro_Customer_Data.csv')  #Reading the saved CSV File into another dataframe
data.tail(5)

# 2) Data Cleaning: Clean the dataset by handling missing values, duplicates, and any inconsistencies.
data.isnull().sum() #Checking number of null values
data.dropna(inplace=True) #Dropping Null values as they will provide in accurate analysis
data.isnull().sum()       #Confirming no more null values present

#Only customer ID should be unique hence only that column should be checked for duplicates
duplicates = data.duplicated(subset='CustomerID',keep=False)
data[duplicates] #Printing duplicated values

#Dropping duplicated values
data=data.drop_duplicates(subset=['CustomerID'])

#Confiriming no more duplicates in CustomerID column left
duplicates = data.duplicated(subset='CustomerID',keep=False)
data[duplicates] #Printing duplicated values

data.describe() #Checking for any inconsistencies
#As visible above the min value for Age is -40 which is an inconsistent value & should be removed
data=data.loc[data['Age']>0] #Removing all negative age values
data.describe()              #Confirming no more inconsistent values left

data.value_counts('Gender') #Further checking for inconsistent values in columns with categories

data.value_counts('PreferredCategory')

data.value_counts('MaritalStatus')  
#As visible above there is an inconsistent value 'status' in MaritalStatus column & should be removed
data=data.loc[data['MaritalStatus']!='status']
data.value_counts('MaritalStatus')  #Confirming removal

len(data)  #Confiriming dataset size after cleaning

data.info()

data.to_csv('TechElectro_Customer_Cleaned.csv',index=False) #Saving cleaned data in csv file
