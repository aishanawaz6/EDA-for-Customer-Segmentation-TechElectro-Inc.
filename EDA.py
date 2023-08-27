# 4) Exploratory Data Analysis: Utilize Python libraries (e.g., Pandas, Matplotlib, Seaborn) to visualize 
# and explore the data, uncovering patterns and insights about customer behavior.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

customerS=pd.read_csv('TechElectro_Customer_Data_Preprocessed.csv')   #Reading saved file
customerS.head()
customerS.describe()


# Visualizing Gender distribution
plt.figure(figsize=(6, 4))
X=customerS['Gender_Male'].value_counts()
plt.pie(x=X,labels=['Male','Female'],colors=['blue','Pink'],autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()


# Visualizing Annual Income distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=customerS, x='AnnualIncome (USD)', kde=True, bins=30,color='Green')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (USD)')
plt.show()


# Visualizing Total Purchases distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=customerS, x='TotalPurchases', kde=True, bins=30,color='maroon')
plt.title('Total Purchases Distribution')
plt.xlabel('Total Purchases')
plt.show()


# Visualizing Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=customerS, x='Age', kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

# Visualizing MaritalStatus distribution 
counts = [customerS['MaritalStatus_Single'].sum(),customerS['MaritalStatus_Divorced'].sum(),customerS['MaritalStatus_Married'].sum()]
plt.pie(counts, labels=['Single', 'Divorced','Married'],colors = sns.color_palette("RdGy", n_colors=3),autopct='%1.1f%%')
plt.title('Marital Status Distribution')
plt.tight_layout()
plt.show()


# Visualizing Preferred Category distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=customerS, x='PreferredCategory_Electronics', hue='PreferredCategory_Appliances',palette='viridis')
plt.title('Preferred Category Distribution')
plt.xlabel('Preferred Category')
plt.legend(title='Appliances', loc='upper right', labels=['Electronics', 'Appliances'])
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(customerS.corr(), annot=True, cmap='RdGy', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Pairplot for numerical features
sns.pairplot(customerS[['Age', 'AnnualIncome (USD)', 'TotalPurchases']])
plt.suptitle('Pairplot for Numerical Features', y=1.02)
plt.show()


# Boxplot for Annual Income across Preferred Category
plt.figure(figsize=(10, 6))
sns.boxplot(data=customerS, x='PreferredCategory_Electronics', y='AnnualIncome (USD)',palette='hls',showmeans=True)
plt.title('Annual Income by Preferred Category')
plt.xlabel('Preferred Category')
plt.ylabel('Annual Income (USD)')
plt.gca().set_xticklabels(['Appliances','Electronics'])
plt.show()

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


