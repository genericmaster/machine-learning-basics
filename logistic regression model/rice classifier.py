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


Multiplot = plt.figure()

#Multiplot.add_subplot(2,2,1).scatter(x=Rice_Data["Major_Axis_Length"],y=Rice_Data["Minor_Axis_Length"])
#Multiplot.add_subplot(2,2,2).scatter(x=Rice_Data["Area"],y= Rice_Data["Eccentricity"])
#Multiplot.add_subplot(2,2,3).scatter(x=Rice_Data["Convex_Area"],y=Rice_Data["Perimeter"])
#Multiplot.add_subplot(2,2,4).scatter(x=Rice_Data["Perimeter"],y=Rice_Data['Extent'])

#plt.show()


#normalizing the dataset

#print(Rice_Data)

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
Encoder =encode()
Class_Encoder=Encoder.fit_transform(Rice_Data["Class"])
Multiplot.add_subplot(2,2,1).scatter(x=Rice_Data["Major_Axis_Length"],y=Rice_Data["Minor_Axis_Length"],c=Class_Encoder)
Multiplot.add_subplot(2,2,2).scatter(x=Rice_Data["Area"],y= Rice_Data["Eccentricity"],c=Class_Encoder)
Multiplot.add_subplot(2,2,3).scatter(x=Rice_Data["Convex_Area"],y=Rice_Data["Perimeter"],c=Class_Encoder)
Multiplot.add_subplot(2,2,4).scatter(x=Rice_Data["Perimeter"],y=Rice_Data['Extent'],c=Class_Encoder)
plt.show()


#splitting the data
Helper = Normalized_Rice_Data.drop(columns="Class")
temp_x_train,Rice_x_test,temp_y_train,Rice_y_test =sp.train_test_split(Helper,Normalized_Rice_Data["Class"],test_size=0.2,train_size=0.8,stratify=Normalized_Rice_Data["Class"],random_state=42)

Rice_x_train,Rice_x_validation,Rice_y_train,Rice_y_validation = sp.train_test_split(temp_x_train,temp_y_train,test_size=0.25,stratify=temp_y_train,random_state=42)




