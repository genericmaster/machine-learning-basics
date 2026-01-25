import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

#data extraction
Taxi_data= pd.read_csv(r"C:\Users\user\Downloads\chicago_taxi_train (1).csv")

#data understanding
Taxi_data.shape
Taxi_data.head(10)
Taxi_data.dtypes

# sanity check
Taxi_data.isna().sum().T
Taxi_data.duplicated().sum()#0 duplicates

#EDA
#descriptive stats
Taxi_data.describe()

#data visualization
#histogram
#for i in Taxi_data.select_dtypes(include="number").columns:
    #list[i]
    #sns.histplot(data=Taxi_data,x=i)
   # plt.show()

#boxplots
#for i in Taxi_data.select_dtypes(include="number").columns:
    #list[i]
    #sns.boxplot(data=Taxi_data,x=i)
    #plt.show()

#scatterplot
#for i in Taxi_data.select_dtypes(include="number").columns:
    #list[i]
    #sns.scatterplot(data=Taxi_data,x=i,y='FARE')#TRIP MILES TRIP SECONDS , TRIP TOTAL
    #plt.show()


#CORRELATION
#corr=Taxi_data.corr(numeric_only=True)
#sns.heatmap(corr,annot=True)
#plt.show()

#information gain
for i in Taxi_data.select_dtypes(include='number').columns:
    if Taxi_data[i].isnull().any() :
        Taxi_data.fillna({i:Taxi_data[i].mean()},inplace=True)

features= Taxi_data.select_dtypes(include="number").drop(columns=('FARE')).copy()
info_gain=mutual_info_regression(X=features,y=Taxi_data['FARE'])
pd.Series(data=info_gain,index=features.columns)

#Normalizing the dataset
def Norm(df:pd.DataFrame) :
    for i in df.select_dtypes(include="number").columns:
        df[i]=df[i].apply(lambda x: (x-df[i].min())/(df[i].max()-df[i].min()))

    return df
    
Norm(Taxi_data)

#splitting data
Temp_x,X_test,Temp_y,Y_test= train_test_split(Taxi_data.drop(columns='FARE'),Taxi_data["FARE"],test_size=0.2,random_state=42)
X_train,X_Val,Y_train,Y_val =train_test_split(Temp_x,Temp_y,test_size=0.25,random_state=42)


#building the model