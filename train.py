import cv2
import os
import numpy as np
from keras import models
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

#parameters
batch_size = 400
epochs = 20
save_dir = './checkpoints'
model_name = 'trained_model.h5'

#count data size
def __getnum__(path):
    fm=os.listdir(path)
    i=0
    for n in fm:
        i+=1
    return i  

#data pre-processing
def __data_label__(path,count): 
    data = np.empty((count,185*594),dtype="float32")
    label = np.empty((count,5),dtype="uint8")
    i=0;
    filename= os.listdir(path)
    for f in filename :
        img = cv2.imread(path+'/'+f,0)
        arr = np.asarray(img,dtype="float32")
        arr = arr.reshape(1,185*594)
        data[i,:] = arr/255
        for j in range(5):
            label[i,j]=int(f[j])
        i+=1
    label = to_categorical(label, 10)
    label = label.reshape(count,50)
    return data,label

#establish the model
model = models.Sequential()

#first layer, dense, 64 nodes
model.add(Dense(64,input_shape=(185*594,)))
model.add(Activation('relu'))#ReLU activation function
model.add(Dropout(0.5)) #50% dropout
#second layer, dense, 64 nodes
model.add(Dense(64))
model.add(Activation('relu'))#ReLU activation function
model.add(Dropout(0.5)) #50% dropout
#third layer, dense, 64 nodes
model.add(Dense(64))
model.add(Activation('sigmoid'))#Sigmoid activation function to uniform output
model.add(Dropout(0.5)) #50% dropout
#output layer, dense, 50 outputs
model.add(Dense(50))
model.add(Activation('softmax'))#Uniform the output into probabilities(0-1)

path = './data'
count=__getnum__(path)
images,labels= __data_label__(path, count)

#split data into train and test sets at the proportion of 7:3
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=30)

#compile the model with Mean Squre Error loss function, RMSprop optimizer
model.compile(loss='MSE', optimizer='rmsprop', metrics=['accuracy'])

#save checkpoints
filepath="model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc',verbose=1,save_best_only=True)
history = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=1,validation_split=0.1)
print(history.history.keys())

#save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
 
#score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

