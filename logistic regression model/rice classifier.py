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
Rice_Data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")
print(Rice_Data)

#data exploration
Range = Rice_Data["Area"].max()-Rice_Data["Area"].min()
Rice_Count = Rice_Data.Class.value_counts(sort=True)
Rice_Data_Distribution= Rice_Data.describe()
Rice_Data.info()
sns.pairplot(data=Rice_Data,y_vars='density',kind="hist",vars=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'])
#plt.show()

#feature selection
Encoder = encode()
Class_encoded=Encoder.fit_transform(Rice_Data['Class'])
Info_Gain = mutual_info_classif(X=Rice_Data.drop(columns='Class'),y=Class_encoded,discrete_features=True)

# data visualisation
Plot_df=Rice_Data.copy()
sns.pairplot(data=Plot_df,vars=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'],hue='Class')
#plt.show()

#normalizing the dataset
def Norm(df:pd.DataFrame) :
    for i in df.select_dtypes(include="number").columns:
        df[i]=df[i].apply(lambda x: (x-df[i].min())/(df[i].max()-df[i].min()))

    return df

Normalized_Rice_Data = Norm(Rice_Data)

#splitting the data
Helper = Normalized_Rice_Data.drop(columns="Class")
temp_x_train,Rice_x_test,temp_y_train,Rice_y_test =sp.train_test_split(Helper,Normalized_Rice_Data["Class"],test_size=0.2,train_size=0.8,stratify=Normalized_Rice_Data["Class"],random_state=42)

Rice_x_train,Rice_x_val,Rice_y_train,Rice_y_val = sp.train_test_split(temp_x_train,temp_y_train,test_size=0.25,stratify=temp_y_train,random_state=42)



# creating the model
def CreateModel() :
    inputs= keras.Input(shape=(7,))
    outputs = keras.layers.Dense(units=1,activation="sigmoid")(inputs)
    Model= keras.Model(inputs=inputs,outputs=outputs)
    metric = [keras.metrics.binary_accuracy,keras.metrics.Precision,keras.metrics.Recall,keras.metrics.AUC]
    Model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.07),loss="binary_crossentropy",metrics=metric)
    
    return Model

#Training the model
def TrainModel( model,Feature_dataframe:pd.DataFrame,Label_dataframe:pd.DataFrame,BatchSize,epoch,Features=[]) :
             
             #logs
           
              
             #early stopping
             Early_Stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    mode= "auto" ,
                    min_delta=1e-4,      
                    restore_best_weights=True       
             )
             
             #training 

             inside = Feature_dataframe[Features].values
             Encoder= encode()
             train_y = Encoder.fit_transform(Label_dataframe)
             val_y = Encoder.transform(Rice_y_val)

             Train =model.fit(x=inside,y=train_y,validation_data=(Rice_x_val,val_y),batch_size=BatchSize,epochs=epoch,shuffle=True,callbacks=[Early_Stopping])
             epoch_history = Train.epoch
             metric_hist = pd.DataFrame(Train.history)
             return (epoch_history,metric_hist)
            

model_1 = CreateModel()

experiment_1 = TrainModel(model_1,Rice_x_train,Rice_y_train,50,500,['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length','Eccentricity', 'Convex_Area', 'Extent'])

#zgraph= testing_model.predict(Rice_x_test).flatten()
#predicted = pd.DataFrame(graph,columns=['predicted_value'])
#df= pd.concat([predicted,Rice_y_test],axis =1)
#sns.scatterplot(y='Class',x='predicted_value', data=df,hue='Class' )

#plt.show()

def TestModel(features,label) :  
     encoding = encode()
     label_test = encoding.fit_transform(label)

     return model_1.evaluate(x=features,y=label_test,batch_size=50)

test1= TestModel(Rice_x_test,Rice_y_test)

print(test1)