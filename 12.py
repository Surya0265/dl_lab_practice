import tensorflow as tf  
import numpy as np 


noisy = np.array([[1,0,1,1,1,1,1,0,1]])
clean= np.array([[1,1,1,1,0,1,1,1,1]])

def train(lr):
    model=tf.keras.Sequential([tf.keras.layers.Dense(5,activation="sigmoid",input_shape=(9,)),tf.keras.layers.Dense(9,activation="sigmoid")])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss="mse")
    h=model.fit(noisy,clean,epochs=10,verbose=1)
    pred=model.predict(noisy)
    loss=h.history['loss'][-1]

    print(np.round(pred))
    print(loss)

train(0.001)
train(0.0001)