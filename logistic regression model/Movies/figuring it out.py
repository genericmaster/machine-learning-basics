import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import keras


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
print(Movies.describe().T)
Movies.describe(include="object").T

#visualization
#histogram to understand distribution of the data
#for i in Movies.select_dtypes(include='number').columns :
    #fig1= plt.figure()
    #fig1= sns.histplot(data=Movies,x=i)
    #plt.show()

#boxplots for finding outliers
#for i in Movies.select_dtypes(include='number').columns :
    #sns.boxplot(data=Movies,x=i)
    #plt.show()
     
#scatterplot to check for relationships
#sns.pairplot(data=Movies,x_vars=['popularity','vote_count'],y_vars=['vote_average'])
#plt.show()

#corr matrix

corr= Movies.corr(numeric_only=True)
#print(corr)
#sns.heatmap(data=corr)
#plt.show()

#data cleaning
Movies.drop(columns=['id','title','overview'],inplace=True)
Movies.rename(columns={'vote_average':'average_rating'},inplace=True)
Movies['release_date']=Movies['release_date'].fillna('2026-01-01')          
Movies['release_date']=pd.to_datetime(Movies['release_date'],errors='raise')
today = pd.Timestamp.now()
Movies['average_vote_day'] = (Movies['vote_count'])/((today-Movies['release_date']).dt.days)
Movies.drop(columns=['release_date'],inplace=True)
#dropping duplicates 
Movies.drop_duplicates(inplace=True)

#data manipulation

#transform
def transform(column,df:pd.DataFrame):
    transform = PowerTransformer(method='yeo-johnson')
    array=transform.fit_transform(df)
    data= pd.DataFrame(data=array,columns=column)
    return data

inf_gain = mutual_info_regression(Movies,Movies['popularity'])
print(inf_gain)  
Movies_data =transform(['popularity','average_rating','vote_count','average_vote_day'],Movies)
for i in Movies_data.select_dtypes(include='number').columns :
    fig1= plt.figure()
    fig1= sns.histplot(data=Movies_data,x=i)
    plt.show()
sns.pairplot(data=Movies_data,x_vars=['average_rating','vote_count','average_vote_day'],y_vars=['popularity'])
plt.show()
inf_gain2 = mutual_info_regression(Movies_data,Movies_data['popularity'])
print(inf_gain2)

#data splitting
x_train,X_test,Y_train,Y_test = train_test_split(Movies_data.drop(columns=['popularity']),Movies_data['popularity'],test_size=0.2,random_state=42,shuffle=True)

def createmodel():
    Inputs = keras.Input(shape=(2,))
    Outputs =keras.layers.Dense(units=1,activation='linear',kernel_initializer='glorot_uniform',bias_initializer='zeros')(Inputs)
    model = keras.Model(inputs=Inputs,outputs=Outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.RootMeanSquaredError()])
    return model

def trainModel(df:pd.DataFrame,features,label,model,batch_size,epoch):
        early_stopping=keras.callbacks.EarlyStopping(
            monitor='val_loss',
         mode='min',
         patience =5,
            min_delta=1e-06,
            restore_best_weights=True
        )
        inputs = df[features].values,
        outputs = label
        train = model.fit(x=inputs,y=outputs,validation_split=0.25,batch_size=batch_size,epochs=epoch,callbacks=[early_stopping])

        return train.history

model_1 = createmodel()
experiment_1 = trainModel(x_train,['vote_count','average_vote_day'],Y_train,model_1,64,500)


