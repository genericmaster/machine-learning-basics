import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#data understanding

# we want to know the shape of our dataset and if all values in the dataset are filled

Rice_Data = pd.read_csv(r"C:\Users\user\Downloads\Rice_Cammeo_Osmancik.csv")

print(Rice_Data.shape) #3810  by 10
print(Rice_Data.columns)#number of columns
print(Rice_Data.dtypes)# data tyoe  of each column
print(Rice_Data.isna().sum())#  ccan see for each column how many null values we have

#drop = Rice_Data.dropna(axis=0)# you can drop nan rows o values
#reset the index every time you drop things

#subsetting a dataframe

Rice_Data_Subset=Rice_Data[['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length']].copy() # makes sure pyton know its a new dataframe


print(Rice_Data.shape)

# feature understanding (univariate analysis)

 #1. run value count function theat accounts for u=duplicates in the series   



Rice_Data['Class'].value_counts()

#2 plotting 
#ax=Rice_Data['Class'].value_counts().plot(kind='bar')
#ax.set_xlabel('type of rice')
#ax.set_ylabel('number of rice')
#fig2= plt.figure()
#fig2=Rice_Data['Area'].plot(kind="density")




# feature relationships

#1 do scatteer plots using seaborn
#scatter= plt.figure()
#scatter = sns.scatterplot(x='Area',y='Perimeter',data=Rice_Data, hue='Class')

#plt.show()
#comapring more than 2 features

sns.pairplot(data=Rice_Data,vars=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
       'Eccentricity', 'Convex_Area', 'Extent'],hue='Class')
#heatmap
corr = plt.figure()
corr = Rice_Data.drop(columns=('Class')).corr()

sns.heatmap(corr,annot=True)

#plt.show()