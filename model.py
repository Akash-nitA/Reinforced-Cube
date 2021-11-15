import tensorflow as tf 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

def model(): 
    input=tf.keras.layers.Input(shape=(None,3))
    hidden=tf.keras.layers.Dense(10,activation='relu',kernel_initializer='he_uniform')(input)
    hidden=tf.keras.layers.Dense(10,activation='relu',kernel_initializer='he_uniform')(hidden)
    output=tf.keras.layers.Dense(6,activation='softmax')(hidden)

    m=tf.keras.Model(inputs=input,outputs=output)
    m.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def preprocessing(train_x,test_x,train_y,test_y):

    train_x=np.array(train_x/255)
    test_x=np.array(test_x/255)
    
    le=LabelEncoder()
    train_y=le.fit_transform(train_y)
    test_y=le.transform(test_y)
    return train_x,test_x,train_y,test_y
    

def training(train_x,test_x,train_y,test_y): 
    train_x,test_x,train_y,test_y=preprocessing(train_x,test_x,train_y,test_y)
    train_y=tf.one_hot(train_y,depth=6)
    test_y=tf.one_hot(test_y,depth=6)

    m=model()
    history=m.fit(train_x,train_y,epochs=10,validation_data=[test_x,test_y])
    m.save('colors.h5')
    plt.plot(history)
    plt.show()
    return history


if __name__ == '__main__':

    df=pd.read_csv('color_data.csv')
    labels=df.pop('label')
    data=df
    train_x,test_x,train_y,test_y=train_test_split(data,labels,test_size=0.2,random_state=1)
    # print(train_x.shape ,test_x.shape,train_y.shape,test_y.shape)
    print(training(train_x,test_x,train_y,test_y))
