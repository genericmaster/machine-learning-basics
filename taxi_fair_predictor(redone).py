#data manipulation
import numpy as np
import pandas as pd

#machine learning
import tensorflow as tf
import keras 

# data visualisation
import matplotlib.pyplot as plt

#dataset
Taxi_fare = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
numeric_features = Taxi_fare.select_dtypes(include=['float64', 'int64'])
Taxi_fare= numeric_features


#Corr_Features = Taxi_fare.corr(numeric_only=True)

#print(Corr_Features)

#CREATING THE MODEL

def CreateModel():
    
   Inputs=keras.Input(shape=(1,))
   first_layer=keras.layers.Dense(units=4,activation="linear")(Inputs)
   second_layer=keras.layers.Dense(units=2,activation="linear")(first_layer)
   Outputs=keras.layers.Dense(units=1,activation="linear")(second_layer)
   Model = keras.Model(inputs=Inputs,outputs=Outputs)
   Model.compile(keras.optimizers.RMSprop(learning_rate=0.001),loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.RootMeanSquaredError()])

   return Model

def trainModel(df:pd.DataFrame,name: str,Label_name:str,Batch_size :int, Epoch:int,model:keras.Model,Feature_name=[]) :
                 
                label = df[Label_name].values
                features=  df[Feature_name].values
                train=model.fit(x=features,y=label,batch_size=Batch_size,epochs=Epoch)
                Epoch_hist=train.epoch
                metrics_history=pd.DataFrame(train.history)
                     
                return( name, Epoch_hist,model, metrics_history)


#testing the model
Taxi_fare["TRIP_MINUTES"]= Taxi_fare["TRIP_SECONDS"]/60

model_1 = CreateModel()

experiment_1 = trainModel(Taxi_fare,"FIRST MODEL","FARE",50,20,model_1,['TRIP_MILES'])

#plotting
y_pred=model_1.predict(Taxi_fare["TRIP_MILES"]).flatten()
fig= plt.subplots(1,1,figsize=(10,8)) 
line,= plt.plot([Taxi_fare["FARE"].min(),Taxi_fare["FARE"].max()],[Taxi_fare["FARE"].min(),Taxi_fare["FARE"].max()])
fig.scatter(x=Taxi_fare["FARE"],y=y_pred,c="green")
fig.add_line(line,)
fig.set_xlabel("actual fare")
fig.set_ylabel("predicteed fare")
plt.show()



#validating model

def BuildBatch(df:pd.DataFrame,batch_size):
        batch = df.sample(n=batch_size).copy()
        batch.set_index(np.arange(batch_size),inplace=True)
        return batch

def PredictFare(model,df,features,label,batch_size=50):
        batch = BuildBatch(df,batch_size)
        predicted_values=model.predict_on_batch(batch[features])

        data={"predicted_fare":[],"observed_fare":[],"L1_loss":[],features:[]}

        for i in range(batch_size):
                predicted= predicted_values[i]
                observed=batch.at[i,label]
                data["predicted_fare"].append(predicted)
                data["observed_fare"].append(observed)
                data["L1_loss"].append(abs(observed-predicted))
                data[features].append(batch.at[i,features])
         
            
            
        output_df = pd.DataFrame(data)
        return output_df


def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

output = PredictFare( model_1,Taxi_fare,"TRIP_MILES", 'FARE')
show_predictions(output)
    




