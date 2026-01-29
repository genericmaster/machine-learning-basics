import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import math 

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

#dropping unnecesarry features
Taxi_data = Taxi_data[['TRIP_SECONDS','TRIP_MILES','FARE','TIPS','TRIP_TOTAL']]


#Normalizing the dataset
def Norm(df:pd.DataFrame) :
    for i in df.select_dtypes(include="number").columns:
        df[i]=df[i].apply(lambda x: (x-df[i].min())/(df[i].max()-df[i].min()))

    return df
    
Norm(Taxi_data)
#splitting data
Temp_x,X_test,Temp_y,Y_test= train_test_split(Taxi_data.drop(columns=['FARE','TRIP_SECONDS','TIPS']),Taxi_data["FARE"],test_size=0.2,random_state=42,shuffle=True)
X_train,X_Val,Y_train,Y_val =train_test_split(Temp_x,Temp_y,test_size=0.25,random_state=42,shuffle=True)

#building the model
def Inputs(df:pd.DataFrame) :
   array=np.array(df)
   if array.ndim<=1:
    array = array.reshape(-1,1)
   else:
       array=array
  
   return array

def Batch(batch_size, X_train, Y_train, X_val, Y_val):
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    for i in range(0, n_train, batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_Y = Y_train[i:i+batch_size]

        # Make sure val batch matches batch size
        start_val = i
        end_val = min(i + batch_size, n_val)   # <-- handles last partial batch
        val_X = X_val[start_val:end_val]
        val_Y = Y_val[start_val:end_val]

        if batch_X.shape[0] == 0 or batch_Y.shape[0] == 0:
            continue
        if val_X.shape[0] == 0 or val_Y.shape[0] == 0:
            continue

        yield batch_X, batch_Y, val_X, val_Y


def Train_Val(batch_gen,Learning_rate):
     np.random.seed(42)
     Weights=np.random.standard_normal(size=(2,1))
     Bias =np.random.standard_normal(1)
     train_loss_track=[]
     val_loss_track=[]
     for epoch in range(0,500):
          size =0
          val_avg =0
          size2 =0
          avg=0
          for batch_X ,batch_Y ,Val_X ,Val_Y in batch_gen:
            train_output = (batch_X @ Weights)+Bias 
            Error = train_output-batch_Y
            Mse_loss=np.mean((Error)**2)
            Rmse_loss = np.sqrt(Mse_loss)
            Weights = Weights - (Learning_rate*((2/len(batch_X))*(batch_X.T@Error)))
            Bias = Bias - Learning_rate*((2/len(batch_X))*(np.sum(Error)))
            size2 +=len(batch_X)
            avg =avg+np.sum(Error**2)
            val_output= (Val_X @ Weights)+Bias
            val_error = val_output - Val_Y
            val_Mse = np.mean((val_error)**2)
            val_rmse = np.sqrt(val_Mse)
            size += len(Val_Y)
            val_avg = val_avg+np.sum(val_error**2)
          Val_loss = np.sqrt(val_avg/size)
          Train_loss=np.sqrt(avg/size2)
          train_loss_track.append(Train_loss)
          val_loss_track.append(Val_loss)
          batch_gen = Batch(128,Inputs(X_train),Inputs(Y_train),Inputs(X_Val),Inputs(Y_val))
     return Rmse_loss, Train_loss,Val_loss,Weights,Bias,train_loss_track,val_loss_track



loss, epoch,val_Loss,Weights,Bias,train_loss,Val_loss = Train_Val(Batch(128,Inputs(X_train),Inputs(Y_train),Inputs(X_Val),Inputs(Y_val)),0.001)  
        
print(f' epoch_train_loss is {epoch} and val_loss is {val_Loss} ' )


def TestBatch(x,y,batch_size):
    for i in range(0,len(x),batch_size):
        yield x[i:i+batch_size],y[i:i+batch_size]


def Test(batch_gen):
            size =0
            total_error =0
            for test_x,test_y in batch_gen:
                output = (test_x@Weights)+Bias
                error = output-test_y
                Mse_loss = np.mean((error)**2)
                Rmse_loss = np.sqrt(Mse_loss)
                total_error+=np.sum(error**2)
                size +=len(test_x)
            test_loss=np.sqrt(total_error/size)
            return Rmse_loss,test_loss


test_loss,epoch_loss =Test(TestBatch(Inputs(X_test),Inputs(Y_test),128))
     
print(f'epoch_test_ loss is {epoch_loss}')

epoches= range(len(train_loss))
plt.figure()
plt.plot(epoches,train_loss,label='Training rmse')
plt.plot(epoches,Val_loss,label='validation rmse')
plt.xlabel('epoch')
plt.ylabel('rmse')
plt.legend()
plt.show()
     



          
