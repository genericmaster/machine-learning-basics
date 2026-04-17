import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import keras

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
for i in Taxi_data.select_dtypes(include="number").columns:
    list[i]
    sns.histplot(data=Taxi_data,x=i)
    plt.show()

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

#dropping unnecesarry features
Taxi_data = Taxi_data[['TRIP_SECONDS','TRIP_MILES','FARE','TIPS','TRIP_TOTAL']]


#Normalizing the dataset
def Norm(df:pd.DataFrame) :
    for i in df.select_dtypes(include="number").columns:
        df[i]=df[i].apply(lambda x: (x-df[i].min())/(df[i].max()-df[i].min()))

    return df
    
Norm(Taxi_data)
#splitting data
Temp_x,X_test,Temp_y,Y_test= train_test_split(Taxi_data.drop(columns=['FARE','TRIP_SECONDS','TIPS']),Taxi_data["FARE"],test_size=0.2,shuffle=True)
X_train,X_Val,Y_train,Y_val =train_test_split(Temp_x,Temp_y,test_size=0.25,shuffle=True)


def CreateModel():
    
   Inputs=keras.Input(shape=(2,))
   Outputs=keras.layers.Dense(units=1,activation="linear",kernel_initializer='glorot_uniform',bias_initializer='zeros')(Inputs)
   Model = keras.Model(inputs=Inputs,outputs=Outputs)
   Model.compile(keras.optimizers.RMSprop(learning_rate=0.0003),loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.RootMeanSquaredError()])

   return Model

def trainModel(df:pd.DataFrame,Label_name:str,Batch_size :int, Epoch:int,model:keras.Model,Feature_name=[]) :
                early_stopping=keras.callbacks.EarlyStopping(
                       monitor='val_root_mean_square',
                       patience= 2,
                       mode= 'auto',
                       min_delta= 1e-06,
                       restore_best_weights=True
                )
                       
                label = df.values
                features=  df[Feature_name].values
                train=model.fit(x=features,y=label,batch_size=Batch_size,epochs=Epoch,validation_data=(X_Val,Y_val),callbacks=[early_stopping])
                metrics_history=pd.DataFrame(train.history)
                metrics_history = metrics_history[['val_root_mean_squared_error','root_mean_squared_error']]
                     
                return( model, metrics_history)


model_1 = CreateModel()

experiment_1,metric_hist = trainModel(X_train,Y_train,64,100,model_1,['TRIP_MILES','TRIP_TOTAL'])

plt.figure()
plt.plot(metric_hist['val_root_mean_squared_error'],label='val_rmse')
plt.plot(metric_hist['root_mean_squared_error'],label='rmse')
plt.xlabel('epoches')
plt.ylabel('rmse')
plt.legend()
plt.show()

#plotting
#y_pred=model_1.predict(Taxi_data[["TRIP_MILES","TRIP_TOTAL"]]).flatten()
#fig= plt.figure().add_subplot(1,1,1) 
#line,= plt.plot([Taxi_data["FARE"].min(),Taxi_data["FARE"].max()],[Taxi_data["FARE"].min(),Taxi_data["FARE"].max()])
#fig.scatter(x=Taxi_data["FARE"],y=y_pred,c="green")
#fig.add_line(line,)
#fig.set_xlabel("actual fare")
#fig.set_ylabel("predicteed fare")





def test(x_test,y_test):
      return model_1.evaluate(x_test,y_test,128)


print(test(X_test,Y_test))








    




