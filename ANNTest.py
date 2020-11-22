import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
#import the data
node_0 = pd.read_csv(r"Prepared Data/Data_Prepared_node0.csv",sep=";")

x_train = node_0.drop("Running App ID",axis=1)
y_train = node_0.loc[:,"Running App ID"]

y_train = y_train.astype('category')

# defining model    
print('Defining Model')

model = Sequential()

model.add(Dense(12, activation="relu", input_shape=(len(x_train[0]),)))
model.add(Dense(120, activation="sigmoid"))
model.add(Dense(49, activation="relu"))
model.add(Dense(7, activation="softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

print('Training Data Node0')
model.fit(
    x_train.reshape(len(x_train), len(x_train[0])),
    y_train,
    epochs=10,
    batch_size=5)
