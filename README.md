# Detect-Pneumonia-Using-X-Ray-Images-with-CNNs-and-Transfer-Learning-Dataquest-Project

## Project Description 
The project is focused on developing a cutting edge technology to assist hospitals in diagnosis of Pheneumonia patients, particularly children. Convolutional Neural Network(CNN) and transfer learning techniques are used on a X-ray images dataset to develop a AI model which can accurately classify the presence of Pheneumonia in children.

## Background 

Image classification in medical diagnostics,with the use of AI and machine learning techniques, has become a pivotal step in current disease diganostics and preventive medicine . It enables accurate, efficient, and automated analysis of medical imaging, significantly enhancing the capabilities of healthcare professionals. Medical image classification is important in early detection of diseases, increased diagnostic accuracy and faster diagnosis, resulting clinicians and physicians making informed decisions in disease diagnostics.

When using various types of medical images such as X-ray, Magnetic Resonance Imaging(MRI), Computed Tomography(CT) scan in image classification, there are various challenges and considerations such as that could impact the classification accuracy of the AI model.High-quality labeled datasets/images,using unbiased datasets ensuring the ethical concerns and fairness are two of the major challenges. 

## Data
A [Mendely Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) formed by researchers from UCSD, collecting chest X-ray images from children, split between training and testing data is used in this project.

### Import the Data File 
[X-ray Dataset](Data/xray_dataset.tar.gz) was uncompressed using the below code: 

```python
import tarfile
def extract_tar_gz(file_path,output_path):
  with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall(path=output_path)

extract_tar_gz('xray_dataset.tar.gz','xray_dataset')
```
### Loading the Dataset 

The **xray_dataset** file is loaded into three subsets of data as `training`, `validation` and `testing` datasets. The original training data was split with a **validation_split** of 0.2 to form the training and validation datasets . The below code shows how the training dataset is loaded. In the similar way validation and testing datasets were laoded. 

``` python
train_set=tf.keras.utils.image_dataset_from_directory(
    directory='xray_dataset/chest_xray/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256,256),
    validation_split=0.2,
    subset='training',
    seed=42,
    shuffle=True
)
```
Each element of the dataset is a tuple in the form of (x,y) in which x denotes a x-ray image as the input feature and y denotes a vector value representing the presence or absence of Phenumonia in the patient as the output. 

## Exploratory Data Analysis (EDA) 

The follwoing python libraries are used when loading, preprocessing and in model building using the preprocessed data. 
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models,Model,Input,applications,initializers
```

There are few insights made from the EDA : 

- All three datasets were rescaled to have a pixel value of 256 and normalized to be in a range of [0,1] using a normalization layer before the EDA of data.
- The images in each datasets were stored in form of batches, so that one batch from each traning and testing datasets were evaluated to gain insights from the x-ray images.
- According to [training image](Images/xray_1.png) and [testing image](Images/xray_2.png) the image labels are one-hot encoded for the classification task, where categorcial classes, Class 0  : [1. 0.] and Class 1  : [0. 1.] reprsents "Absence" and "Presence" of Phenuemonia in that particular image.
- Having 3 chanels in the shape of the image shows these images are RGB images although they are appeared to be greyscaled images or have been converted to RGB by replicating the single channel across the three color channels.
- Observing the training and testing images in the first batch doesn't carry much information due to overlapping of the ribs and lungs in the image.
- So, that data augmentation methods such as adding rotations, improving the contrast of the image would be helpful in extracting the features specific to Phneumonia infection.

## Model Building 

### 1. CNN model 

After building a simple CNN model, the model initially resulted a high validation accuracy of 97.7% representing the potential  overfitting of the model. So,that regularization,data augmentation and more fully connected layers are added to the model to prevent overfitting as shown in the below code: 

```python
model=models.Sequential()

model.add(layers.RandomZoom(height_factor=0.1,fill_mode='reflect'))
model.add(layers.RandomRotation(factor=0.2))
model.add(layers.RandomContrast(factor=0.2))


model.add(layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',
                     padding='same',strides=2,input_shape=(256,256,3)))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2))

model.add(layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',
                     padding='same',strides=2))
model.add(layers.MaxPooling2D(pool_size=(3,2),strides=2))
model.add(layers.Flatten())


model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(2))


early_stopping_callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

opt=tf.keras.optimizers.Adam(learning_rate=0.001)
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])

model.fit(normalized_train_set,validation_data=normalized_validation_set,
           epochs=10,callbacks=[early_stopping_callback])
```

### 2. Transfer Learning Model

Transfer Learning is the method of creating a new model by using a pre-trained optimized model on a similar dataset : 
- By freezing weights of the pre-trained model to extract features for the current dataset
- Fine tuning the model on the current dataset by unfreezing all or only particular layers  of the pre-trainde model's architecture.

This method helps to conserve the rich feature reprsentations learned by the already trained model, improoving the training speed and accuracy of the transfer learned model. Tensorflow's ResNet based pre-trained model **resnet_v2.ResNet50V2** trained on ImageNet dataset is used as the pre-trained model to create the transfer learning model using the current x-ray dataset for this project.

## Model Evaluation 

### 1. CNN model 

- After experimenting and modifying the model with number of connected layers, their units and adding regularization technqiues the optimal model resulted a validation accuracy of 90.6% after training only for 10 epochs.
- The testing accuracy of the model is 85.9% which indicates the model is pretty generalized and a fairly good value.

### 2. Transfer Learning model 
- The validation accuracy of the model was 90.6% when the base model was trained for 10 epochs after freezing weights of all layers when computing features.
- When only the last 10 layers of the model was frozen and after being trained for 5 epochs, the resulted validation accuracy of the model was 97.4% which indicates high overfitting of the model.
- Somehow the testing accuracy of the model was decreased to 62.5% , representing a poor model for practial use.

## Conclusion and Furute Work 

- Using transfer learning technique saves lots of training time, but due to certain factors such as potentially a smaller size of the current dataset, inbalaced classes, improper freezing or unfreezing  layers wiyhout understanding the structure of the pre-trained model, inadequate data augmentation resulted a poor testing accuracy.
- So, that further study on the structure architecture of the pre-trained model and the resemblence of imageNet dataset used to train the base model is needed before proceeding with further experimentation on the fine tuning of the transfer learning model.
- CNN model performed fairly well, although its not an excellent model to be used in a practical setting for the detection of the Phenumonia in disease diagnostics, it can be used in preliminary studies to gain an idea about how to detect the condition.
- Both models need further improvements in its performance such as by adding more regularization and data augmenttaion methods to improve the overall test accuracy in order to be confidently use in an real-world hospital setting for disease diagnosis.  







