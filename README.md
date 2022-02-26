# Brain_tumor_prediction_using_cnn
## Brain Tumor
A brain tumor, known as an intracranial tumor, is an abnormal mass of tissue in which cells grow and multiply uncontrollably, seemingly unchecked by the mechanisms that control normal cells. More than 150 different brain tumors have been documented, but the two main groups of brain tumors are termed primary and metastatic.
## Problem
Brain tumors are complex to detect as there are many abnormilities in the shape, sizes and location of the brain tumor(s), Which makes really difficult to understand the nature of the tumor.Developing countries lacks in skillful doctors and lack of knowledge about tumors makes it challenging and time-consuming to make conclusion from MRI.
So, to predict it and classify tumor using, Convolutional Neural Network and Transfer Learning is the game changing asset of deep learning.

## Dataset
It is a kaggle data comprises of 3 dataset in the given problem that are test,train and validation data set.The test and train dataset comprises of 2 classes yes and no.The shape of the train dataset is around 3000 combining both the class and test dataset comprises of 1600 records of both class.
You can get the dataset from the below link.
https://drive.google.com/drive/folders/1Zr8O4ad2ZxcT_BycKxFO4vmNzSHy-n7h

## Approach
The model designed for the given problem has following definition:
1-->model = Sequential();
model.add(Conv2D(32, (3, 3), input_shape=input_shape));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(2, 2)));
model.add(Conv2D(32, (3, 3)));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(2, 2)));
model.add(Conv2D(64, (3, 3)));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(2, 2)));
model.add(Flatten());
model.add(Dense(64));
model.add(Activation('relu'));
model.add(Dropout(0.5));
model.add(Dense(1));
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics = ['accuracy'])
To decrease the training time, I have resized the input images to 150x150, so input image shape is 64x64x3.
2-->To compile the loss I used Binary_crossentropy and the optimizer used is rmsprop because the centered version additionally maintains a moving average of the gradients, and uses that average to estimate the variance.
![image](https://user-images.githubusercontent.com/99955096/155849291-de07d4b3-cd37-4715-b986-c822fd079ffd.png)

## Result
From the above graph we can clearly see that the training loss is decreasing accross the iteration and  the accuracy of the model is increasing over the period of time.The accuracy of the model is 97% utmost and the loss for the given accuracy is 0.0948.
