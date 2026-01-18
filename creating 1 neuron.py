
from keras import Sequential
from keras.layers import Dense


# sequential - a way to build a neural network layer by layer 
#dense - a fully connected layer 
model = Sequential([
    Dense(
        units =1,
        activation = "linear",
        kernel_initializer ="glorot_uniform",
        bias_initializer= "zeros",
        input_shape =(1,)


    )

])
