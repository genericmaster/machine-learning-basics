#data manipulation
import pandas as pd
import numpy as np

#deep learning
import keras
from  sklearn.model_selection import _split as sp
from sklearn.preprocessing import LabelEncoder as encode

#data exploration 
import matplotlib.pyplot as plt



Rice_Data = pd.read_csv(r"C:\Users\user\Downloads\Rice_Cammeo_Osmancik.csv")

Minimum= Rice_Data["Major_Axis_Length"].min()

Maximum= Rice_Data["Major_Axis_Length"].max()

Range = Rice_Data["Area"].max()-Rice_Data["Area"].min()

Std_From_Mean= (Rice_Data["Perimeter"].max() -Rice_Data["Perimeter"].mean())/Rice_Data["Perimeter"].std()

#print("Minumum Major_Axis_Length: ",Minimum,"\n","Maximum Major_Axis_Length :",Maximum,"\n","Area Range:",Range,"\n","perimeter Std_From_Mean :",Std_From_Mean)
  
  
# data visualisation


#class colouring
Encoder =encode()
Class_Encoder=Encoder.fit_transform(Rice_Data["Class"])

#plots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0,0].scatter(Rice_Data["Major_Axis_Length"], Rice_Data["Minor_Axis_Length"], c=Class_Encoder)
axs[0,0].set_xlabel("Major_Axis_Length")
axs[0,0].set_ylabel("Minor_Axis_Length")

axs[0,1].scatter(Rice_Data["Area"], Rice_Data["Eccentricity"], c=Class_Encoder)
axs[0,1].set_xlabel("Area")
axs[0,1].set_ylabel("Eccentricity")

axs[1,0].scatter(Rice_Data["Convex_Area"], Rice_Data["Perimeter"], c=Class_Encoder)
axs[1,0].set_xlabel("Convex_Area")
axs[1,0].set_ylabel("Perimeter")

axs[1,1].scatter(Rice_Data["Perimeter"], Rice_Data["Extent"], c=Class_Encoder)
axs[1,1].set_xlabel("Perimeter")
axs[1,1].set_ylabel("Extent")

plt.tight_layout()
#plt.show()

#normalizing the dataset

def Norm(Rice_Data):
    
    Rice_Data["Area"]= Rice_Data["Area"].apply(lambda x : (x-Rice_Data["Area"].min())/(Rice_Data["Area"].max()-Rice_Data["Area"].min()))
    Rice_Data["Perimeter"]= Rice_Data["Perimeter"].apply(lambda x : (x-Rice_Data["Perimeter"].min())/(Rice_Data["Perimeter"].max()-Rice_Data["Perimeter"].min()))
    Rice_Data["Major_Axis_Length"]= Rice_Data["Major_Axis_Length"].apply(lambda x : (x-Rice_Data["Major_Axis_Length"].min())/(Rice_Data["Major_Axis_Length"].max()-Rice_Data["Major_Axis_Length"].min()))
    Rice_Data["Minor_Axis_Length"]= Rice_Data["Minor_Axis_Length"].apply(lambda x : (x-Rice_Data["Minor_Axis_Length"].min())/(Rice_Data["Minor_Axis_Length"].max()-Rice_Data["Minor_Axis_Length"].min()))
    Rice_Data["Eccentricity"]= Rice_Data["Eccentricity"].apply(lambda x : (x-Rice_Data["Eccentricity"].min())/(Rice_Data["Eccentricity"].max()-Rice_Data["Eccentricity"].min()))
    Rice_Data["Convex_Area"]= Rice_Data["Convex_Area"].apply(lambda x : (x-Rice_Data["Convex_Area"].min())/(Rice_Data["Convex_Area"].max()-Rice_Data["Convex_Area"].min()))
    Rice_Data["Area"]= Rice_Data["Extent"].apply(lambda x : (x-Rice_Data["Extent"].min())/(Rice_Data["Extent"].max()-Rice_Data["Extent"].min()))

    return Rice_Data


Normalized_Rice_Data = Norm(Rice_Data)

#print(Normalized_Rice_Data)

#splitting the data
Helper = Normalized_Rice_Data.drop(columns="Class")
temp_x_train,Rice_x_test,temp_y_train,Rice_y_test =sp.train_test_split(Helper,Normalized_Rice_Data["Class"],test_size=0.2,train_size=0.8,stratify=Normalized_Rice_Data["Class"],random_state=42)

Rice_x_train,Rice_x_validation,Rice_y_train,Rice_y_validation = sp.train_test_split(temp_x_train,temp_y_train,test_size=0.25,stratify=temp_y_train,random_state=42)



# creating the model

def CreateModel() :

    inputs= keras.Input(shape=(2,))
    outputs = keras.layers.Dense(units=1,activation="sigmoid")(inputs)
    Model= keras.Model(inputs=inputs,outputs=outputs)
    metric = [keras.metrics.binary_accuracy,keras.metrics.Precision,keras.metrics.Recall,keras.metrics.AUC]
    Model.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0),loss="binary_crossentropy",metrics=metric)
    
    return Model

def TrainModel( model,Feature_dataframe:pd.DataFrame,Label_dataframe:pd.DataFrame,BatchSize,epoch,Features=[]) :
              inside = Feature_dataframe[Features].values
              Encoder= encode()
              classes = Encoder.fit_transform(Label_dataframe)
              
              Train = model.fit(x=inside,y=classes,batch_size=BatchSize,epochs=epoch)
              epoch_history = Train.epoch
              metric_hist = pd.DataFrame(Train.history)
              return (epoch_history,metric_hist)


model_1 = CreateModel()

experiment_1 = TrainModel(model_1,Rice_x_train,Rice_y_train,50,50,["Area","Eccentricity"])