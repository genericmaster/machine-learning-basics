import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
 


#read dataset
Movies = pd.read_csv(r"C:\Users\user\Downloads\top_rated_movies.csv")
Movies.head(5)
Movies.tail(5)
Movies.dtypes
#sanity check
Movies.shape
Movies.isnull().sum()
Movies.isnull().value_counts()
Movies.duplicated().value_counts()
Movies.info()

#EDA
#descriptive statistics
Movies.describe().T
Movies.describe(include="object").T

#visualization
#histogram to understand distribution of the data
for i in Movies.select_dtypes(include='number').columns :
    fig1= plt.figure()
    fig1= sns.histplot(data=Movies,x=i)
    #plt.show()

#boxplots for finding outliers
for i in Movies.select_dtypes(include='number').columns :
    sns.boxplot(data=Movies,x=i)
    #plt.show()
     
#scatterplot to check for relationships
sns.pairplot(data=Movies,x_vars=['vote_count','vote_average'],y_vars=['popularity'])
#plt.show()

#corr matrix

corr= Movies.corr(numeric_only=True)
#print(corr)
sns.heatmap(data=corr)
#plt.show()

side =pd.read_csv(r"C:\Users\user\Downloads\california_housing_train.csv")
print(side.describe())