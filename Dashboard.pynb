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
