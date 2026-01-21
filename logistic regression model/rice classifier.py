#data manipulation
import pandas as pd
import numpy as np

#deep learning
import keras
from  sklearn.model_selection import _split as sp
from sklearn.preprocessing import LabelEncoder as encode
from sklearn.feature_selection import mutual_info_classif
#data exploration 
import matplotlib.pyplot as plt
import seaborn as sns

#dataset
Rice_Data = pd.read_csv(r"C:\Users\user\Downloads\Rice_Cammeo_Osmancik.csv")

#data exploration
Range = Rice_Data["Area"].max()-Rice_Data["Area"].min()
Rice_Count = Rice_Data.Class.value_counts(sort=True)
Rice_Data_Distribution= Rice_Data.describe()
Rice_Data.info()
sns.pairplot(data=Rice_Data,y_vars='density',kind="hist",vars=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'])
plt.show()

#feature selection
Encoder = encode()
Class_encoded=Encoder.fit_transform(Rice_Data['Class'])
Info_Gain = mutual_info_classif(X=Rice_Data.drop(columns='Class'),y=Class_encoded,discrete_features=True)

# data visualisation
Plot_df=Rice_Data.copy()
sns.pairplot(data=Plot_df,vars=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'],hue='Class')
plt.show()

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

#splitting the data
Helper = Normalized_Rice_Data.drop(columns="Class")
temp_x_train,Rice_x_test,temp_y_train,Rice_y_test =sp.train_test_split(Helper,Normalized_Rice_Data["Class"],test_size=0.2,train_size=0.8,stratify=Normalized_Rice_Data["Class"],random_state=42)

Rice_x_train,Rice_x_validation,Rice_y_train,Rice_y_validation = sp.train_test_split(temp_x_train,temp_y_train,test_size=0.25,stratify=temp_y_train,random_state=42)

# creating the model
def CreateModel() :
    inputs= keras.Input(shape=(7,))
    outputs = keras.layers.Dense(units=1,activation="sigmoid")(inputs)
    Model= keras.Model(inputs=inputs,outputs=outputs)
    metric = [keras.metrics.binary_accuracy,keras.metrics.Precision,keras.metrics.Recall,keras.metrics.AUC]
    Model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.8),loss="binary_crossentropy",metrics=metric)
    
    return Model

#Training the model
def TrainModel( model,Feature_dataframe:pd.DataFrame,Label_dataframe:pd.DataFrame,BatchSize,epoch,Features=[]) :
              inside = Feature_dataframe[Features].values
              Encoder= encode()
              classes = Encoder.fit_transform(Label_dataframe)
              Train = model.fit(x=inside,y=classes,batch_size=BatchSize,epochs=epoch)
              epoch_history = Train.epoch
              metric_hist = pd.DataFrame(Train.history)
              return (epoch_history,metric_hist,"Training")
            

model_1 = CreateModel()

experiment_1 = TrainModel(model_1,Rice_x_train,Rice_y_train,50,20,['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'])

#validation data
validation_experiment_1 = TrainModel(model_1,Rice_x_validation,Rice_y_validation,64,64,['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'])

#testing data
testing_experiment_1 = TrainModel(model_1,Rice_x_test,Rice_y_test,64,64,['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'])

#graph= testing_model.predict(Rice_x_test).flatten()
#predicted = pd.DataFrame(graph,columns=['predicted_value'])
#df= pd.concat([predicted,Rice_y_test],axis =1)
#sns.scatterplot(y='Class',x='predicted_value', data=df,hue='Class' )

#plt.show()