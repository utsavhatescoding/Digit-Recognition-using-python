Implementing Simple Neural Networks

 #Code Starts From Here :

 import cv2 as cv;
  import numpy as np; 
  import matplotlib.pyplot as plt;
   import tensorflow as tf;

mnist=tf.keras.datasets.mnist;
 (X_train, Y_train),(X_test, Y_test)=mnist.load_data();

X_train=tf.keras.utils.normalize(X_train, axis=1);
 X_test=tf.keras.utils.normalize(X_test, axis=1);

model=tf.keras.models.Sequential();
 model.add(tf.keras.layers.Flatten(input_shape=(28,28)));
  model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu));
   model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu));
    model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax));

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy') ;
model.fit(X_train, Y_train,epochs=3);

accuracy, loss=model.evaluate(X_test,Y_test)

model.save('Mero Model')

#Now, I will feed in my own handwritten digits to this system' Mero Model '

img = cv.imread(f'{6}.png')[:, :,0] img = np.invert(np.array([img])) prediction = model.predict(img) print(f'I GUESS THE NUMBER IS : {np.argmax(prediction)} ') plt.imshow(img[0],cmap=plt.cm.binary) plt.show()

#I have images of handwritten digits made with Ms-Paint.
#{ and right curly bracket } #[0]