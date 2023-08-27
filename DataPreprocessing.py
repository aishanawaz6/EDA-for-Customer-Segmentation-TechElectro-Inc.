import pandas as pd
# 3) Data Preprocessing: Perform feature scaling, normalization, and encode categorical variables if needed.
data=pd.read_csv('TechElectro_Customer_Cleaned.csv') #Reading cleaned data
#Perfoming one-hot encoding on categorical variables [ENCODING]
dataPreprocessed=pd.get_dummies(data,columns=['PreferredCategory','Gender','MaritalStatus'])
dataPreprocessed.head()

#Using min-max scaling on some numerical columns [FEATURE SCALING]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
colsToScale=['AnnualIncome (USD)', 'TotalPurchases']
dataPreprocessed[colsToScale] = scaler.fit_transform(dataPreprocessed[colsToScale])
dataPreprocessed.head() #As visible, further normalization is not required

dataPreprocessed.to_csv('TechElectro_Customer_Data_Preprocessed.csv',index=False) #Saving preprocessed data in csv file
