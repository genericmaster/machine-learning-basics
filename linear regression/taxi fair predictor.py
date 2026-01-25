# DATA
import pandas as pd
import numpy as np
import tensorflow as tf
#ML
import keras
import ml_edu.experiment
import ml_edu.results
# DATA VISALIZATION
import plotly.express as px

# importing cicago taxi fares

Chicago_Taxi_Fare = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

#maximum fare
#print(Chicago_Taxi_Fare["FARE"].max())

#MEAN DISTANCE ACROSS
#print(Chicago_Taxi_Fare["TRIP_MILES"].mean())

#NUMBER OF CAB COMPANIES
#Company_Number = Chicago_Taxi_Fare["COMPANY"].nunique()
#print(Company_Number)

# most frequent payment method
#Payment_Method = Chicago_Taxi_Fare["PAYMENT_TYPE"].mode()
#print(Payment_Method)

#correlation matrix
#corr = Chicago_Taxi_Fare.corr(numeric_only=True)
#print(corr)

#plottin  out the correlation

#@title Code - View pairplot
#fig=px.scatter_matrix(Chicago_Taxi_Fare, dimensions=["FARE", "TRIP_MILES", "TIP_RATE"])
#fig.show()

#training model

def CreateModel(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],
        )->keras.Model:
        inputs ={name:keras.Input(shape=(1,),name=name)for name in settings.input_features}
        concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))# takes the single input layer values and combines them into a full list
        outputs = keras.layers.Dense (units=1)(concatenated_inputs)# creates  one neuron that is dense that will produce a single output
        model = keras.Model(outputs=outputs,inputs=inputs)# where data enters and leaves

        #compiling the model
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate), loss ="mean_squared_error",metrics=metrics)
        return model
def trainMOdel(
                experiment_name: str,
                model:keras.Model,
                dataset : pd.DataFrame,
                label_name: str,
                settings: ml_edu.experiment.ExperimentSettings,
)-> ml_edu.experiment.Experiment:
        
        
 features = {name:dataset[name].values for name in settings.input_features}
 label = dataset[label_name].values
 history = model.fit(x= features,
                     y= label,
                     batch_size=settings.batch_size,
                     epochs=settings.number_epochs)
 

                     

 return ml_edu.experiment.Experiment(
      name = experiment_name ,
      settings=settings,
      model=model,
      epochs =history.epoch,
      metrics_history=pd.DataFrame(history.history), ) 


#training with 1 feature
settinngs_1 =ml_edu.experiment.ExperimentSettings(
     learning_rate= 0.001,
     number_epochs= 20,
     batch_size= 50,
     input_features= ['TRIP_MILES']
)


metrics =[keras.metrics.RootMeanSquaredError(name='rmse')]

#model_1 = CreateModel(settinngs_1,metrics)

#experiment_1 = trainMOdel('one_feature',model_1,Chicago_Taxi_Fare,'FARE',settinngs_1)
#ml_edu.results.plot_experiment_metrics(experiment_1,['rmse'])
#ml_edu.results.plot_model_predictions(experiment_1,Chicago_Taxi_Fare,'FARE')

                

#when we increase the learning rate by 1 the learning rate doesnt really converge to its lowest point , we experinece flactuations in the loss function
#at leraning rate of 0.001 the model converges too slow and we end up having  too many outliers(the prediction line either undershoots or oversoots the data points)
settings_2 = ml_edu.experiment.ExperimentSettings(
     learning_rate=0.001,
     batch_size=50,
     number_epochs=20,
     input_features=['TRIP_MILES',"TRIP_MINUTES"]
)

Chicago_Taxi_Fare['TRIP_MINUTES'] = Chicago_Taxi_Fare['TRIP_SECONDS']/60

metrics=[keras.metrics.RootMeanSquaredError(name ="rmse")]
model_2 = CreateModel(settings_2,metrics)
experimemt_2 = trainMOdel("two_features",model_2,Chicago_Taxi_Fare,'FARE',settings_2)
#ml_edu.results.plot_experiment_metrics(experimemt_2,['rmse'])
#ml_edu.results.plot_model_predictions(experimemt_2,Chicago_Taxi_Fare,'FARE')

#ml_edu.results.compare_experiment([experiment_1, experimemt_2], ['rmse'], Chicago_Taxi_Fare, Chicago_Taxi_Fare['FARE'].values)
#preiction functions

def format_currency(x):
     return "${:,.2f}".format(x)

def build_batch(df,batch_size):
     batch = df.sample(n=batch_size).copy()
     batch.set_index(np.arange(batch_size),inplace=True)
     return batch

def predict_fare(model,df,features,label,batch_size=50):
     batch = build_batch(df,batch_size)
     predicted_values = model.predict_on_batch(x={name:batch[name].values for name in features})
     data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
     for i in range(batch_size):
          predicted = predicted_values[i][0]
          observed=batch.at[i,label]
          data['PREDICTED_FARE'].append(format_currency(predicted))
          
          data["OBSERVED_FARE"].append(format_currency(observed))
          data['L1_LOSS'].append(format_currency(abs(observed-predicted)))
          data[features[0]].append(batch.at[i,features[0]])
          data[features[1]].append("{:.2f}".format(batch.at[i,features[1]]))


     output_df = pd.DataFrame(data)
     return output_df
     
def show_predictions(output):
         header = "-" * 80
         banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
         print(banner)
         print(output)
         return

output = predict_fare(experimemt_2.model, Chicago_Taxi_Fare, experimemt_2.settings.input_features, 'FARE')
show_predictions(output)