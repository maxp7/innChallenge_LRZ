import pandas as pd
import numpy
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import os

from tensorflow.python import keras

root_logdir = os.path.join(os.curdir, "tensorlogs")

def get_run_logdir():
    import time
    run_id = time.strftime(r"run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

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
model.add(Dense(96, activation="relu"))
model.add(Dense(60, activation="relu"))
model.add(Dense(21, activation="relu"))
model.add(Dense(7, activation="softmax"))


model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

tensorbord_cb = keras.callbacks.TensorBoard(root_logdir)

print('Training Data Node0')
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=5,validation_split=0.3,callbacks=[tensorbord_cb])
    
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # set the vertical range to [0-1]
plt.show()