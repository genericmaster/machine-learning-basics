
import numpy as np 
from sklearn.preprocessing import LabelBinarizer
import keras
import string



text = "Digital technology transforms how modern society functions today. Innovative software developers build complex systems using advanced programming languages. These powerful computers process massive datasets to uncover hidden patterns within information. As artificial intelligence evolves, machine learning algorithms become more efficient at solving difficult problems. Continuous education helps professionals stay relevant in a rapidly changing technical landscape. Logical thinking and creative problem solving remain essential skills for every successful engineer. Future breakthroughs in hardware will likely accelerate scientific discovery even further. By understanding these core concepts, students can master the fundamentals of data science and build impactful tools for tomorrow future."
text=text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))
master_list = text.split(sep= " ")

x_train = []
y_train = []

for word in range(0,len(master_list)):
 
#looking foward
    for i in range(word+1, word+3):
        if i< len(master_list):
            x_train.append(master_list[word])
            y_train.append(master_list[i])
#looking backwards
    
    for j in range(word-2,word):
        if j>=0:
            x_train.append(master_list[word])
            y_train.append(master_list[j])



binazer = LabelBinarizer()

master_dictionary = binazer.fit(master_list)
x_train = master_dictionary.transform(x_train).astype('float32')
y_train = master_dictionary.transform(y_train).astype('float32')
vocab_size = x_train.shape[1]
words = binazer.classes_
np.save('vocab.npy', words)

def create_model():
   model = keras.Sequential()
   model.add(keras.Input(shape=(vocab_size,)))
   model.add(keras.layers.Dense(units=100,activation="linear"))
   model.add(keras.layers.Dense(units=vocab_size,activation="softmax"))
   model.compile(optimizer="adam",loss="categorical_crossentropy")

   return model


def train_model(model:keras.Model,x_train,y_train):
        train = model.fit(x=x_train,y=y_train,epochs =200 ,batch_size= 8)
        weights= model.layers[0].get_weights()[0]
        np.save('word_vectors.npy', weights)


model_1 = create_model()

output_model = train_model(model_1,x_train,y_train) 
print(output_model)
