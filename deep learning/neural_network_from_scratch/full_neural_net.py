import numpy as np
import math 
np.random.seed(42)
f' this script handles both foward and backpropagation so that the model actually learns from data, added optimizers include sgd,momentum,rmsprop,adam'

class Neural_net:
   def __init__(self,loss):
      self.add_layer_list = [] 
      self.output_layer = None
      self.loss = loss

   class Layer:
     def __init__(self,neurons,features,activation):
         self.neurons = neurons
         self.features = features
         self.activation = activation
         self.weight_matrix = self.weight_initialization()
         self.bias_matrix = self.bias_initialization()
         self.input = None
         self.z = None
         self.output = None
         
     def weight_initialization(self):
          if self.activation in ['sigmoid', 'tanh','softmax']:
            weight_matrix = np.random.uniform(low= -math.sqrt(6/(self.features+self.neurons)),high=math.sqrt(6/(self.features+self.neurons)),size=(self.features,self.neurons))
          elif self.activation == "linear":
            weight_matrix = np.random.randn(self.features,self.neurons)
          elif self.activation =="relu":
            weight_matrix = np.random.uniform(low =-math.sqrt(2/self.features), high= math.sqrt(2/self.features),size=(self.features,self.neurons))
          return weight_matrix
       
     def bias_initialization(self):
            bias_matrix = np.zeros(shape=(1,self.neurons))
            return bias_matrix
       
     def forward(self,input):
         self.input = input
         linear_trans = np.dot(a=input,b=self.weight_matrix)
         linear_trans = linear_trans + self.bias_matrix
         self.z = linear_trans
         if self.activation == 'linear':
            self.output = linear_trans
            return linear_trans
         if self.activation =='sigmoid':
            sigmoid = 1/(1+np.exp(-linear_trans) )
            self.output = sigmoid
            return sigmoid
         if self.activation =="tanh":
            tanh =  (np.exp(linear_trans)-np.exp(-linear_trans))/(np.exp(linear_trans)+np.exp(-linear_trans))
            self.output =tanh
            return tanh
         if self.activation =="relu":
            relu = np.maximum(linear_trans,0)
            self.output =relu
            return relu
         
     def forward_output_layer(self,input,activation):
         self.input = input
         linear_trans = np.dot(a=input,b=self.weight_matrix)
         linear_trans = linear_trans + self.bias_matrix
         self.z = linear_trans
         if activation in ['linear','sigmoid','softmax']:
            if activation== 'linear':
               self.output = linear_trans
               return linear_trans
            if activation== 'sigmoid':
               sigmoid = 1/(1+np.exp(-linear_trans))
               self.output = sigmoid
               return sigmoid
            if activation =="softmax":
               if self.neurons>1:
                  max = np.max(linear_trans,axis=1,keepdims=True)
                  softmax = np.exp(linear_trans-max)/np.sum(np.exp(linear_trans-max),axis=1, keepdims=True) #subtract max to avoid  number overflow
                  self.output = softmax
                  return softmax
               else:
                  raise ValueError("neurons must be set>1")
         else:
            raise ValueError("specify required activation from linear , sigmoid and softmax")
         
     def backward_output_layer(self,labels,loss):
        if loss in ['mse','binary_cross_entropy','categorical_cross_entropy']:
           delta = (self.output-labels)/self.output.shape[0]
           gradient = np.dot(self.input.T ,delta)
           passed_back = delta @ self.weight_matrix.T
           return passed_back,gradient,delta
     def backward_prop(self,incoming_gradient):
        if self.activation == 'linear':
            delta = incoming_gradient * 1
            gradient = self.input.T @ delta
            passed_back = delta @ self.weight_matrix.T
            return passed_back, gradient,delta
        if self.activation == "sigmoid":
            delta = incoming_gradient * (self.output * (1 - self.output))
            gradient = self.input.T @ delta
            passed_back = delta @ self.weight_matrix.T
            return passed_back, gradient,delta
        if self.activation == 'tanh':
            delta = incoming_gradient*(1- (self.output)**2)
            gradient = self.input.T@delta
            passed_back =(delta @ self.weight_matrix.T)
            return passed_back,gradient,delta
        
        if self.activation == 'relu':
            delta = incoming_gradient * np.where(self.z > 0, 1, 0)
            gradient = self.input.T @ delta
            passed_back = delta @ self.weight_matrix.T
            return passed_back, gradient,delta
        else:
              incoming =0
              delta =0
              gradient =0
              return incoming ,gradient,delta         
                      
   def add_layer(self,neurons,features,activation):
      layer = Neural_net.Layer(neurons, features, activation)
      self.add_layer_list.append(layer)
      
   def result_layer(self,neurons,features,activation):
      self.output_activation = activation
      self.output_layer = Neural_net.Layer(neurons,features,self.output_activation)
               
   def forward_prop(self,user_data):
      current = user_data
      for layer in  self.add_layer_list:
        current=layer.forward(current)
      current = self.output_layer.forward_output_layer(current,self.output_activation)
      return current
     
   def loss_function(self,prediction,true_labels):
      if self.loss == 'mse':
         mse = (1/prediction.shape[0])*np.sum((prediction-true_labels)**2)
         return float(mse)
      if self.loss == 'binary_cross_entropy':
         if all(p in [0,1] for p in true_labels):
            binary_cross_entropy = -(1/prediction.shape[0]) * np.sum(true_labels*np.log(prediction) + (1-true_labels)*np.log(1-prediction))
         else:
            raise ValueError("binary_cross_entropy only requires inputs of 0 or 1")
         return float(binary_cross_entropy)
      if self.loss == "categorical_cross_entropy":
          if np.array_equal(np.unique(true_labels), [0,1]):
            categorical_cross_entropy = -(np.sum(np.sum(true_labels*np.log(np.clip(prediction, 1e-7, 1)),axis=1))/prediction.shape[0])
            return float(categorical_cross_entropy)
          else:
            raise ValueError("categorical_cross_entropy only requires inputs of 0 or 1")
         
   def train(self,X,labels,learning_rate=0.01,epoch =50,optimizer='sgd'):
         loss_track =[]
         velocity_weight = [0]*len(self.add_layer_list)
         velocity_output_layer_weight=0
         velocity_output_layer_bias =0
         velocity_bias = [0]*len(self.add_layer_list)
         velocity_weight_rmsprop =[0]*len(self.add_layer_list)
         velocity_bias_rmsprop =  [0]*len(self.add_layer_list)
         velocity_output_layer_weight_rmsprop=0
         velocity_output_layer_bias_rmsprop=0
         velocity_weight_adam = [0]*len(self.add_layer_list)
         velocity_bias_adam =[0]*len(self.add_layer_list)
         m_weight_adam =[0]*len(self.add_layer_list)
         m_bias_adam =[0]*len(self.add_layer_list)
         m_weight_output_layer_adam =0
         m_bias_output_layer_adam =0
         velocity_output_layer_weight_adam =0
         velocity_output_layer_bias_adam =0
         t=0
         
         for i in range(epoch):
            t=t+1
            prediction = self.forward_prop(X)
            loss_value = self.loss_function(prediction,labels)
            
            loss_track.append(loss_value)
            incoming_gradient,gradient,delta = self.output_layer.backward_output_layer(labels, self.loss)
            output_delta = delta
            output_gradient = gradient
            gradient_descent =[]
               
            for layer in reversed(self.add_layer_list):
               incoming_gradient,gradient,delta = layer.backward_prop(incoming_gradient)
               gradient_descent.append((layer, gradient,delta))
            
            if optimizer =='sgd':   
               for layer, gradient,delta in gradient_descent:
                  layer.weight_matrix = layer.weight_matrix - learning_rate * gradient
                  layer.bias_matrix = layer.bias_matrix - learning_rate * np.sum(delta, axis=0, keepdims=True)
                                    
            #update for output layer
               weight=self.output_layer.weight_matrix
               weight = weight - learning_rate*output_gradient
               self.output_layer.weight_matrix = weight
               bias = self.output_layer.bias_matrix
               bias = bias - learning_rate*np.sum(output_delta,axis=0,keepdims=True)
               self.output_layer.bias_matrix = bias
                      
            if optimizer == 'rmsprop':
               for index, (gradient, v_t,v_t_bias) in enumerate(zip(gradient_descent, velocity_weight_rmsprop,velocity_bias_rmsprop)):
                  v_t = 0.9*v_t +(1-0.9)*(gradient[1]**2)
                  v_t_bias =0.9*v_t_bias +(1-0.9)* (np.sum(gradient[2]**2, axis=0, keepdims=True))
                  gradient[0].weight_matrix=gradient[0].weight_matrix - learning_rate * (gradient[1]/np.sqrt(v_t+1e-08))
                  gradient[0].bias_matrix = gradient[0].bias_matrix - learning_rate*(np.sum(gradient[2], axis=0, keepdims=True)/np.sqrt(v_t_bias+1e-08))
                  velocity_weight_rmsprop[index] = v_t
                  velocity_bias_rmsprop[index]=v_t_bias
                  
               #update for output layer
               weight=self.output_layer.weight_matrix
               velo_t_rmsprop = 0.9*velocity_output_layer_weight_rmsprop+(1-0.9)*(output_gradient**2)
               weight = weight - learning_rate*(output_gradient/np.sqrt(velo_t_rmsprop+1e-08))
               self.output_layer.weight_matrix = weight
               velocity_output_layer_weight_rmsprop=velo_t_rmsprop
               bias = self.output_layer.bias_matrix
               velo_t_bias_rmsprop = 0.9*velocity_output_layer_bias_rmsprop + (1-0.9)*np.sum(output_delta**2,axis=0,keepdims=True)
               bias = bias - learning_rate*(np.sum(output_delta,axis=0,keepdims=True)/np.sqrt(velo_t_bias_rmsprop +1e-08))
               self.output_layer.bias_matrix = bias
               velocity_output_layer_bias_rmsprop=velo_t_bias_rmsprop
            if optimizer == "momentum":
               for index, (gradient, v_t,v_t_bias) in enumerate(zip(gradient_descent, velocity_weight,velocity_bias)):
                  v_t = 0.9*v_t +(1-0.9)*gradient[1]
                  v_t_bias =0.9*v_t_bias +(1-0.9)* np.sum(gradient[2], axis=0, keepdims=True)
                  gradient[0].weight_matrix=gradient[0].weight_matrix - learning_rate * v_t
                  gradient[0].bias_matrix = gradient[0].bias_matrix - learning_rate*v_t_bias
                  velocity_weight[index] = v_t
                  velocity_bias[index]=v_t_bias
                  
               #update for output layer
               weight=self.output_layer.weight_matrix
               velo_t = 0.9*velocity_output_layer_weight+(1-0.9)*output_gradient
               weight = weight - learning_rate*velo_t
               self.output_layer.weight_matrix = weight
               velocity_output_layer_weight=velo_t
               bias = self.output_layer.bias_matrix
               velo_t_bias = 0.9*velocity_output_layer_bias +(1-0.9)*np.sum(output_delta,axis=0,keepdims=True)
               bias = bias - learning_rate*velo_t_bias
               self.output_layer.bias_matrix = bias
               velocity_output_layer_bias=velo_t_bias
            if optimizer== 'adam':
               
               for index,(gradient,v_t_adam,v_t_adam_bias,m_t_adam,m_t_bias_adam) in enumerate (zip(gradient_descent,velocity_weight_adam,velocity_bias_adam,m_weight_adam,m_bias_adam)):
                   #momentum
                   m_t_adam = 0.9*m_t_adam +(1-0.9)*gradient[1]
                   m_t_bias_adam =0.9*m_t_bias_adam +(1-0.9)* np.sum(gradient[2], axis=0, keepdims=True)
                   m_weight_adam[index] = m_t_adam
                   m_bias_adam[index]= m_t_bias_adam
                   
                  #rmsprop
                   v_t_adam = 0.99*v_t_adam +(1-0.99)*(gradient[1]**2)
                   v_t_adam_bias =0.99*v_t_adam_bias +(1-0.99)* (np.sum(gradient[2]**2, axis=0, keepdims=True))
                   velocity_weight_adam[index] = v_t_adam
                   velocity_bias_adam[index] = v_t_adam_bias
                 #bias correction
                   m_t_correction = m_t_adam/((1-(0.9**t)))
                   m_t_bias_adam_corresction = m_t_bias_adam/((1-(0.9**t)))
                   v_t_adam_correction = v_t_adam/((1-(0.99**t)))
                   v_t_adam_bias_correction = v_t_adam_bias/((1-(0.99**t)))
                   
                 #weight update
                   gradient[0].weight_matrix=gradient[0].weight_matrix - learning_rate *(m_t_correction/np.sqrt(v_t_adam_correction+1e-08))
                   gradient[0].bias_matrix = gradient[0].bias_matrix - learning_rate*(m_t_bias_adam_corresction/np.sqrt(v_t_adam_bias_correction+1e-08))
               
               #output layer adam logic
               weight=self.output_layer.weight_matrix
               velocity_output_layer_weight_adam = 0.99*velocity_output_layer_weight_adam+(1-0.99)*(output_gradient**2)
               m_weight_output_layer_adam = 0.9*m_weight_output_layer_adam+(1-0.9)*output_gradient
               
               bias = self.output_layer.bias_matrix
               velocity_output_layer_bias_adam = 0.99*velocity_output_layer_bias_adam + (1-0.99)*np.sum(output_delta**2,axis=0,keepdims=True)
               m_bias_output_layer_adam = 0.9*m_bias_output_layer_adam +(1-0.9)*np.sum(output_delta,axis=0,keepdims=True)
               
               #bias coreection output layer
               velocity_output_layer_weight_adam_corr = velocity_output_layer_weight_adam/((1-(0.99**t)))
               velocity_output_layer_bias_adam_corr = velocity_output_layer_bias_adam/((1-(0.99**t)))
               m_weight_output_layer_adam_corr = m_weight_output_layer_adam/((1-(0.9**t)))
               m_bias_output_layer_adam_corr = m_bias_output_layer_adam/((1-(0.9**t)))
               
              #weight update
               weight= weight - learning_rate *(m_weight_output_layer_adam_corr/np.sqrt(velocity_output_layer_weight_adam_corr+1e-08))
               bias = bias - learning_rate *(m_bias_output_layer_adam_corr/np.sqrt(velocity_output_layer_bias_adam_corr+1e-08))
               self.output_layer.weight_matrix = weight
               self.output_layer.bias_matrix = bias
  
         return loss_track   
               
