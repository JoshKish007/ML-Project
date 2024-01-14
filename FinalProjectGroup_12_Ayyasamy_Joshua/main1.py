import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option('display.max_columns', None) # Show all columns


# # Loading the dataset
import pandas as pd

# Update the file path based on the actual location of your CSV file on macOS
bicycleTheft = pd.read_csv("/Users/joshua/Downloads/Bicycle_Theft_In_Toronto/bicycle-thefts - 4326.csv")


print(bicycleTheft)


# # Data understanding


print(bicycleTheft.info())

print(bicycleTheft.describe())


print(bicycleTheft.dtypes.value_counts())


bicycleTheft_numeric = bicycleTheft.select_dtypes(include='number')

print(bicycleTheft_numeric.corr())

print(bicycleTheft.isnull().sum())


for x in bicycleTheft:
    print(f"Unique Values in {x} : {bicycleTheft[x].nunique()}")


import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# Assuming bicycleTheft is a DataFrame with columns 'PREMISES_TYPE' and 'STATUS'

#chart showing bicyle theft based on type of premises
plt.rcParams['figure.figsize'] = (10, 6)
sns.histplot(x="PREMISES_TYPE", hue="STATUS", data=bicycleTheft, stat="count", multiple="stack")
plt.show()

#chart showing bicyle theft - Reported
plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="REPORT_DOW", hue="STATUS", data=bicycleTheft, stat="count", multiple="stack")
plt.show()

#chart showing bicyle theft - Occurrences by weekdays
plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="OCC_DOW", hue="STATUS", data=bicycleTheft, stat="count", multiple="stack")
plt.show()
#chart showing bicyle theft - Occurrences by hours
plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="OCC_HOUR", hue="STATUS", data=bicycleTheft, stat="count", multiple="stack")
plt.show()
#chart showing bicyle theft - Histogram Frequency count
plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="BIKE_TYPE", hue="STATUS", data=bicycleTheft, stat="count", multiple="stack")
plt.show()

statusOfBicycle = bicycleTheft.groupby(['BIKE_TYPE','STATUS', 'OCC_YEAR']).size().reset_index(name='count')

statusOfBicycle01 = statusOfBicycle[statusOfBicycle['STATUS'] == 'STOLEN']

print(statusOfBicycle01)

plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="BIKE_TYPE", hue="STATUS", data=statusOfBicycle01, stat="count", multiple="stack")
plt.show()

plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="OCC_YEAR" ,hue="STATUS", data=statusOfBicycle, stat="count", multiple="stack")
plt.show()

plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="OCC_YEAR" ,hue="STATUS", data=statusOfBicycle01, stat="count", multiple="stack")
plt.show()

plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="OCC_DOY", hue="STATUS", data=bicycleTheft, stat="count", multiple="stack")
plt.show()

plt.rcParams['figure.figsize']=(10,6)
sns.histplot(x="REPORT_DOY", hue="STATUS", data=bicycleTheft, stat="count", multiple="stack")
plt.show()

# ## Histograms for frequency count of each column

bicycleTheft.hist(figsize = (15,10))
plt.show()

plt.figure(figsize = (15,8))
sns.pairplot(bicycleTheft)
plt.show()

#Coorelation Matrix
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(bicycleTheft_numeric.corr(), annot=True,linewidths=.5, ax=ax)
plt.show()

# # Data Preparation

# ## 1. Removing the unnecessary columns:-

bicycleTheft01 = bicycleTheft.drop(['_id', 'EVENT_UNIQUE_ID', 'DIVISION', 'geometry','OCC_DATE', 'OCC_DOY','LOCATION_TYPE',
                   'REPORT_DATE', 'REPORT_MONTH', 'REPORT_YEAR', 'REPORT_DOY','REPORT_HOUR', 'BIKE_MODEL',
                   'REPORT_DAY'], axis=1)

print(bicycleTheft01)

bicycleTheft01 = bicycleTheft01[bicycleTheft01['OCC_YEAR'] == 2022]

print(bicycleTheft01)

bicycleTheft01 = bicycleTheft01.drop(['OCC_YEAR'], axis =1)

print(bicycleTheft01)

print(bicycleTheft01.info())


#FILLING NULL VALUES
bicycleTheft01['BIKE_COLOUR'].fillna('Other', inplace=True)
bicycleTheft01['BIKE_MAKE'].fillna('Other', inplace=True)

bicycleTheft01['OCC_HOUR'] = bicycleTheft01['OCC_HOUR'].fillna(bicycleTheft01['OCC_HOUR'].mean())
bicycleTheft01['BIKE_SPEED'] = bicycleTheft01['BIKE_SPEED'].fillna(bicycleTheft01['BIKE_SPEED'].mean())

print(bicycleTheft01.info())


#bicycleTheft01['BIKE_COST'].min()
print(bicycleTheft01['BIKE_COST'].max())


# ## binning
#num_bins = 4
#bin_labels = ['Low', 'Average', 'High', 'Luxury']
#bicycleTheft01[''] = pd.cut(bicycleTheft01['BIKE_COST'], bins=num_bins, labels=bin_labels)
low = bicycleTheft01['BIKE_COST'].quantile(.25)
average = bicycleTheft01['BIKE_COST'].quantile(.5)
high = bicycleTheft01['BIKE_COST'].quantile(.75)
bicycleTheft01['BIKE_COST_CATEGORY'] = np.select(
    [
        bicycleTheft01['BIKE_COST'].isna(),
        bicycleTheft01['BIKE_COST'] <= low,
        (bicycleTheft01['BIKE_COST'] > low) & (bicycleTheft01['BIKE_COST'] <= average),
        (bicycleTheft01['BIKE_COST'] > average) & (bicycleTheft01['BIKE_COST'] <= high),
        bicycleTheft01['BIKE_COST'] > high
    ],
    [
        'NK',
        'Low',
        'Average',
        'High',
        'Luxury'
    ],
    default='Unknown'
)


bicycleTheft01 = bicycleTheft01.drop(['BIKE_COST'], axis =1)

#Performing Encoding of Categorical data
# encoding categorical features
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
encoder = preprocessing.LabelEncoder()
categoricalData = [col for col in bicycleTheft01.columns if bicycleTheft01[col].dtype == 'object']
for col in categoricalData:
    bicycleTheft01[col] = encoder.fit_transform(bicycleTheft01[col])
X, Y = bicycleTheft01.drop('STATUS', axis=1), bicycleTheft01['STATUS']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)



print(bicycleTheft01)


print(x_train)


print(y_train)



print(x_test)


print(y_test)



print(bicycleTheft01.info())


