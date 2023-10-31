import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

data = pd.read_csv('/kaggle/input/historical-hourly-weather-data/city_attributes.csv')
data

data.dtypes

data.shape

data.duplicated()

data['Country'].replace('Israel', 'Palestine', inplace=True)
data
data = data.drop(columns=['Latitude', 'Longitude'])
data

data['Country'].unique()

data['City'].unique()

data2 = pd.read_csv('/kaggle/input/historical-hourly-weather-data/humidity.csv')
data2

data2.shape

data2.info()

data2.head(10)

data2.duplicated()

data2.isnull().sum()

data2.dropna(inplace=True)
data2.isnull().sum()

data2.columns
data2.head()
data2.describe()
data2['datetime'] = pd.to_datetime(data2['datetime'])
data2.info()

data2.nlargest(1,'Vancouver')

