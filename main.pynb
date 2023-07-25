# DataScience Remote Internship - Aisha Nawaz Week 5 Day 01
# Project 1: Exploratory Data Analysis (EDA) for Customer Segmentation

# Client Name: TechElectro Inc.
# Company Name: DataGenius Analytics
# Description: TechElectro Inc. is a prominent electronics retailer with a widespread customer base. 
# They are keen on gaining deeper insights into their customers' preferences and behaviors to optimize their 
# marketing strategies & enhance customer satisfaction.

# DataGenius Analytics has been selected to conduct an exploratory data analysis (EDA) project 
# that will help TechElectro Inc. discover meaningful patterns and segment their customers based on their characteristics.

# Dataset: TechElectro_Customer_Data.csv
# | CustomerID | Age | Gender | MaritalStatus | AnnualIncome (USD) | TotalPurchases | PreferredCategory |
# |------------|-----|--------|---------------|-------------------|----------------|-------------------|
# | 1001       | 33  | Male   | Married       | 65000             | 18             | Electronics       |
# | 1002       | 28  | Female | Single        | 45000             | 15             | Appliances        |
# | 1003       | 42  | Male   | Single        | 55000             | 20             | Electronics       |
# | 1004       | 51  | Female | Married       | 80000             | 12             | Electronics       |
# | 1005       | 37  | Male   | Divorced      | 58000             | 10             | Appliances        |
# | ...        | ... | ...    | ...           | ...               | ...            | ...               |

# (Note: The dataset should contains a total of 500 customers. You can create remaining dataset yourself)
# Description of the columns:
# CustomerID: Unique identifier for each customer.
# Age: Age of the customer.
# Gender: Gender of the customer (Male/Female).
# MaritalStatus: Marital status of the customer (Married/Single/Divorced).
# AnnualIncome: Annual income of the customer in USD.
# TotalPurchases: Total number of purchases made by the customer.
# PreferredCategory: The category of products the customer prefers (e.g., Electronics, Appliances).

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

customerS=pd.read_csv('TechElectro_Customer_Data_Preprocessed.csv')   #Reading saved file
customerS.head()

# 4) Exploratory Data Analysis: Utilize Python libraries (e.g., Pandas, Matplotlib, Seaborn) to visualize 
# and explore the data, uncovering patterns and insights about customer behavior.
import matplotlib.pyplot as plt
import seaborn as sns
customerS.describe()

# Visualizing Gender distribution
plt.figure(figsize=(6, 4))
X=customerS['Gender_Male'].value_counts()
plt.pie(x=X,labels=['Male','Female'],colors=['blue','Pink'],autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()

As visible in the pie chart above, The company has roughly the same percentage of male & female customers.
There is a slight difference of 2 % with more males than females but nothing that strongly suggests
that TechElectro Inc must target any gender more than the other.

# Visualizing Annual Income distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=customerS, x='AnnualIncome (USD)', kde=True, bins=30,color='Green')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (USD)')
plt.show()

The chart clearly shows that customers annual income vary greatly and is completely random. It suggests that TechElectro Inc's customers do not have customers that belong to a specific range of income (Eg. high, low). Hence they do not need to target any specific class of people.

# Visualizing Total Purchases distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=customerS, x='TotalPurchases', kde=True, bins=30,color='maroon')
plt.title('Total Purchases Distribution')
plt.xlabel('Total Purchases')
plt.show()

TechElectro Inc seem to have customers with total purchases varying a lot. However, There are many customers with high number of purchases suggesting that the company's products are doing well for now. 

# Visualizing Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=customerS, x='Age', kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

TechElectro Inc customer's age vary but more customers are around 30 years old and very less are around 90. TechElectro Inc' products seem to target more people in their 30s.

# Visualizing MaritalStatus distribution 
counts = [customerS['MaritalStatus_Single'].sum(),customerS['MaritalStatus_Divorced'].sum(),customerS['MaritalStatus_Married'].sum()]
plt.pie(counts, labels=['Single', 'Divorced','Married'],colors = sns.color_palette("RdGy", n_colors=3),autopct='%1.1f%%')
plt.title('Marital Status Distribution')
plt.tight_layout()
plt.show()

There's not a lot of difference between marital status of TechElectro Inc's customers. However more married people preffer buying TechElectro Inc's products than single or divorced.

# Visualizing Preferred Category distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=customerS, x='PreferredCategory_Electronics', hue='PreferredCategory_Appliances',palette='viridis')
plt.title('Preferred Category Distribution')
plt.xlabel('Preferred Category')
plt.legend(title='Appliances', loc='upper right', labels=['Electronics', 'Appliances'])
plt.show()

TechElectro Inc's customers seem to prefer buying more from the category appliances than electronics. TechElectro Inc's can work on improving their products in the electronics category as well.

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(customerS.corr(), annot=True, cmap='RdGy', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

There seems to be no strong correlation between any of the attributes. TechElectro Inc's customers do not have any strong pattern visible, they are completely random. 

# Pairplot for numerical features
sns.pairplot(customerS[['Age', 'AnnualIncome (USD)', 'TotalPurchases']])
plt.suptitle('Pairplot for Numerical Features', y=1.02)
plt.show()

The figure above shows the randomness of TechElectro Inc's customers. This can be a good thing for the company. They seem to attract all sorts of customers rather than focusing on a specific type.

# Boxplot for Annual Income across Preferred Category
plt.figure(figsize=(10, 6))
sns.boxplot(data=customerS, x='PreferredCategory_Electronics', y='AnnualIncome (USD)',palette='hls',showmeans=True)
plt.title('Annual Income by Preferred Category')
plt.xlabel('Preferred Category')
plt.ylabel('Annual Income (USD)')
plt.gca().set_xticklabels(['Appliances','Electronics'])
plt.show()

Customer's buying from electronics section seem to have a slightly high annual income on average than appliances section customers. TechElectro Inc may increase their prices in this section.

# Violin plot for Age by Gender & Preferred Category
plt.figure(figsize=(10, 6))
sns.violinplot(data=customerS, x='Gender_Male', y='Age', hue='PreferredCategory_Electronics', split=True, palette='cividis')
plt.title('Age Distribution by Gender and Preferred Category')
plt.xlabel('Gender')
plt.ylabel('Age')

# Setting custom legend labels & colors
customLegendLabels = ['Appliances', 'Electronics']
customLegendColors = sns.color_palette('cividis', n_colors=2)  

# Creating custom legend
legendPatches = [plt.Line2D([], [], marker='o', linestyle='-', markersize=8, markerfacecolor=color) for color in customLegendColors]
plt.legend(legendPatches, customLegendLabels, title='Preferred Category', loc='upper right')

plt.gca().set_xticklabels(['Female', 'Male'])  

plt.show()

Female customers of TechElectro Inc seem to preffer more electronics than appliances whereas Male customers prefer both the categories roughly the same.

# 5) Customer Segmentation: Apply clustering algorithms (e.g., K-means) to segment customers based on their buying patterns,
# demographics, and preferences.
#Tools: Python, Jupyter Notebook, Pandas, Matplotlib, Seaborn, Scikit-learn

from sklearn.cluster import KMeans
import warnings

# To Suppress some warnings i was getting related to KMeans memory leak issue
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Selecting relevant features for clustering
features = ['Age', 'AnnualIncome (USD)', 'TotalPurchases','PreferredCategory_Electronics','PreferredCategory_Appliances']
customerFeatures=customerS[[feature for feature in features]]

# Determining the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(customerFeatures)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method to Find Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-cluster Sum of Squares (Inertia)')
plt.xticks(range(1, 11))
plt.show()

Optimal value of k seems to be 5

# Elbow Method above shows the optimal k value to be equal to 5
k = 5

# Applying K-means clustering 
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
customerS['Cluster'] = kmeans.fit_predict(customerFeatures)

# Visualizing the clusters using a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=customerS, x='Age', y='TotalPurchases', hue='Cluster', palette='dark', s=80)
plt.title('Customer Segmentation - K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Total Purchases')
plt.legend(title='Cluster', loc='upper right')
plt.show()

There seems to be more purchases made by TechElectro Inc's customers with age ranging from 40-60

# Deployment: After completing the EDA and customer segmentation, 
# DataGenius Analytics will create an interactive dashboard using Dash or Streamlit. 
# This dashboard will allow TechElectro Inc. to explore the customer segments and access visualizations that reveal 
# customer preferences, helping them optimize marketing campaigns and tailor their offerings to specific customer groups.
import dash
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_mantine_components as dmc
import matplotlib.pyplot as plt
import seaborn as sns

external_stylesheets = [dmc.theme.DEFAULT_COLORS]
app = Dash(__name__, external_stylesheets=external_stylesheets) #Defining app

#Reading saved data
data=pd.read_csv('TechElectro_Customer_Cleaned.csv') 
data2=pd.read_csv('TechElectro_Customer_Data_Preprocessed.csv')

# Defining the layout of the dashboard
app.layout = html.Div([
    html.H1('Customer Segmentation Dashboard'),
    
    # Pie chart for Gender distribution
    dcc.Graph(
        id='gender-pie-chart',
        figure={
            'data': [{
                'type': 'pie',
                'labels': ['Male', 'Female'],
                'values': data2['Gender_Male'].value_counts(),
                'marker': {'colors': ['blue', 'pink']},
                'hoverinfo': 'label+percent',
                'textinfo': 'percent',
                'textposition': 'inside',
            }],
            'layout': {
                'title': 'Gender Distribution',
                'showlegend': False,
                'height': 400,
            }
        }
    ),

    # Histogram for Annual Income distribution
    dcc.Graph(
        id='annual-income-histogram',
        figure={
            'data': [{
                'type': 'histogram',
                'x': data2['AnnualIncome (USD)'],
                'nbinsx': 30,
                'marker': {'color': 'green'},
            }],
            'layout': {
                'title': 'Annual Income Distribution',
                'xaxis': {'title': 'Annual Income (USD)'},
                'yaxis': {'title': 'Count'},
            }
        }
    ),

    # Histogram for Total Purchases distribution
    dcc.Graph(
        id='total-purchases-histogram',
        figure={
            'data': [{
                'type': 'histogram',
                'x': data2['TotalPurchases'],
                'nbinsx': 30,
                'marker': {'color': 'maroon'},
            }],
            'layout': {
                'title': 'Total Purchases Distribution',
                'xaxis': {'title': 'Total Purchases'},
                'yaxis': {'title': 'Count'},
            }
        }
    ),

    # Histogram for Age distribution
    dcc.Graph(
        id='age-histogram',
        figure={
            'data': [{
                'type': 'histogram',
                'x': data2['Age'],
                'nbinsx': 30,
            }],
            'layout': {
                'title': 'Age Distribution',
                'xaxis': {'title': 'Age'},
                'yaxis': {'title': 'Count'},
            }
        }
    ),

    # Pie chart for Marital Status distribution
    dcc.Graph(
        id='marital-status-pie-chart',
        figure={
            'data': [{
                'type': 'pie',
                'labels': ['Single', 'Divorced', 'Married'],
                'values': [data2['MaritalStatus_Single'].sum(), data2['MaritalStatus_Divorced'].sum(), data2['MaritalStatus_Married'].sum()],
                'marker': {'colors': sns.color_palette("RdGy", n_colors=3)},
                'hoverinfo': 'label+percent',
                'textinfo': 'percent',
                'textposition': 'inside',
            }],
            'layout': {
                'title': 'Marital Status Distribution',
                'showlegend': False,
                'height': 400,
            }
        }
    ),

    # Countplot for Preferred Category distribution
    dcc.Graph(
        id='preferred-category-countplot',
        figure={
            'data': [{
                'type': 'bar',
                'x': ['Electronics', 'Appliances'],
                'y': [data2['PreferredCategory_Electronics'].sum(), data2['PreferredCategory_Appliances'].sum()],
                'marker': {'color': sns.color_palette("viridis")},
            }],
            'layout': {
                'title': 'Preferred Category Distribution',
                'xaxis': {'title': 'Preferred Category'},
                'yaxis': {'title': 'Count'},
            }
        }
    ),
    
    # Title for the dropdown
    html.Label('Select a category from PreferredCategory:'),

    # Adding Dropdown to allow user to select a category from PreferredCategory
    dcc.Dropdown(
        id='segment-dropdown',
        options=[{'label': segment, 'value': segment} for segment in data['PreferredCategory'].unique()],
        value=data['PreferredCategory'].unique()[0],
        clearable=False
    ),
    
    # Scatter plot to visualize Age vs. Annual Income
    dcc.Graph(id='scatter-plot'),
        # Boxplot for Annual Income across Preferred Category
    dcc.Graph(
        id='boxplot-preferred-category',
        figure={
            'data': [{
                'type': 'box',
                'x': data2['PreferredCategory_Electronics'],
                'y': data2['AnnualIncome (USD)'],
                'marker': {'color': sns.color_palette("hls")},
                'boxmean': True,
            }],
            'layout': {
                'title': 'Annual Income by Preferred Category',
                'xaxis': {'title': 'Preferred Category'},
                'yaxis': {'title': 'Annual Income (USD)'},
            }
        }
    ),
# Violin plot for Age by Gender & Preferred Category
    dcc.Graph(
        id='violin-plot-age-gender-preferred-category',
        figure={
            'data': [{
                'type': 'violin',
                'x': data2['Gender_Male'],
                'y': data2['Age'],
                'box': {'visible': True},
                'points': 'all',
                'jitter': 0.5,
                'pointpos': -1.8,
                'hoverinfo': 'x+y',
                'scalegroup': 'violins',
                'scalemode': 'count',
                'orientation': 'v',
                'name': 'Age Distribution',
            }],
            'layout': {
                'title': 'Age Distribution by Gender and Preferred Category',
                'xaxis': {'title': 'Gender'},
                'yaxis': {'title': 'Age'},
                'showlegend': True,
                'legend': {'title': 'Preferred Category', 'x': 0.85, 'y': 0.95},
            }
        }
    ),
    
])

# Defining the callback to update the scatter plot based on the selected category
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('segment-dropdown', 'value')]
)
def updateScatterPlot(cat): #This plot will change according to preffered category
    filteredData = data[data['PreferredCategory'] == cat]
    fig = px.scatter(filteredData, x='Age', y='AnnualIncome (USD)', color='PreferredCategory', 
                     labels={'Age': 'Age', 'AnnualIncome (USD)': 'AnnualIncome (USD)'})
    fig.update_layout(title=f'Scatter plot for {cat} customers')
    return fig

# Running the app
if __name__ == '__main__':
    app.run(debug=True, port=8090, use_reloader=False)
