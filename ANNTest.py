import pandas as pd
import numpy
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#import the data
node_0 = pd.read_csv(r"Prepared Data/Data_Prepared_node0.csv",sep=";")

x_train = node_0.drop(["Running App ID", "Running App"],axis=1)
y_train = node_0.loc[:,"Running App ID"]

x_train = x_train.to_numpy()
y_train = to_categorical(y_train)

# defining model    
print('Defining Model')

model = Sequential()

model.add(Dense(12, input_shape=(len(x_train[0]),)))
model.add(Dense(120, activation="tanh"))
model.add(Dense(49, activation="relu"))
model.add(Dense(7, activation="softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

print('Training Data Node0')
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=5,validation_split=0.3)
