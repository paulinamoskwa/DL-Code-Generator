###-----------------------------------------------------------------------------------------
### Libraries
###-----------------------------------------------------------------------------------------
import streamlit as st
from bokeh.models.widgets import Div

###-----------------------------------------------------------------------------------------
### Title
###-----------------------------------------------------------------------------------------
if st.button('Hello 👋'):
    js = "window.open('https://paulinomoskwa.github.io/Hello/')"  # New tab or window
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

st.markdown('# Deep Learning for Image Recognition on Google Colab - Code Generator 🥳')
st.markdown("""##### You want to build an artificial neural network to automatically classify images, but you don't feel like *bothering with the code*?""")
st.markdown('''##### Let me do it for you! 🤓''')
st.markdown('''Follow the instructions in the "getting started" section to create the working environment. 
    Select on the sidebar the neural network parameters that inspire you most (or choose the recommended ones). 
    Copy, paste and run the code generated below on this page on a google colab notebook and .. you're done!''')

st.image("https://miro.medium.com/max/1400/1*rAbCk0T4rksShBcPQjWC0A.gif", use_column_width=True) # width=500)

###-----------------------------------------------------------------------------------------
### Getting Started
###-----------------------------------------------------------------------------------------
st.markdown("### Getting started")
st.markdown(''' 
    - Make a Google account and log in to the drive.
    - Import your dataset to the drive in a **.zip** format.\\
      Inside 'dataset.zip' there must be a folder called (again) 'dataset' which contains one sub-folder per category.
      The images must be divided into these sub-folders. Sub-folders names, as well as the images names, 
      are not important, what is important is that the main folder is called "dataset" and that each sub-folder contains 
      images from only one category. The 'dataset.zip' file must have the following structure:
''')

code = '''
# 📂 dataset
#  └── 📂 Category 1
#  |    └── 📄 image_00.jpg
#  |    └── 📄 image_01.jpg   
#  |    └── ..     
#  |    └── 📄 image_99.jpg   
#  |
#  └── 📂 Category 2
#  |    └── 📄 image_00.jpg
#  |    └── 📄 image_01.jpg   
#  |    └── ..     
#  |    └── 📄 image_99.jpg   
#  |
#       ..
#  |
#  └──  📂 Category N
#        └── 📄 image_00.jpg
#        └── 📄 image_01.jpg   
#        └── ..     
#        └── 📄 image_99.jpg   
'''
st.code(code, language='python')

st.markdown('''
    - Create and open a new Google Colab notebook
    - Run the first chunk of code to connect the notebook to Google
''')

code = '''
from google.colab import drive
drive.mount('/content/drive')
'''
st.code(code, language='python')

st.markdown('''
    - Set the neural network parameters on the left column of this page
    - Copy, paste and run the generated code below on your notebook
    - Wait eons for the training to finish.. and you're done! Now you have your classifier!
''')

###-----------------------------------------------------------------------------------------
### Sidebar Parameters
###-----------------------------------------------------------------------------------------

col1, col2, col3 = st.sidebar.columns([1, 15, 1])
with col2:
    st.markdown('## Make your choices! ✍️')
    
    # Image size
    img_h_w = st.selectbox('📌 Select the preferred image size:', ('128', '256 (Recommended)', '512')) 
    if img_h_w == '256 (Recommended)':
        img_h_w = 256
    with st.expander('More info'):
        st.markdown('''The larger the image size, the better the accuracy of the classifier, 
            for the cost of more training time.''')

    # Batch size
    batch_size = st.selectbox('📌 Select the batch size:', ('8', '16', '32 (Recommended)', '64', '128', '256'))
    if batch_size == '32 (Recommended)':
        batch_size = 32

    # Train-validation split
    train_val_split = st.slider('📌 Percentage of data for validation:', 0.2, 1.0)
    with st.expander('More info'):
        st.markdown('Recommended: **0.2** or 0.3')

    # Non-trainable layers
    non_trainable = st.slider('📌 Percentage of non-trainable layers:', 0.1, 1.0)
    with st.expander('More info'):
        st.markdown('''Recommended: <= 0.8\\
            The lower the percentage, the longer the training time.''')

    # Number of epochs
    num_epochs = st.selectbox('📌 Select the number of epochs:', 
        ('1 (Just to test)', '2', '5', '10', '50', '100', '200', '500 (Recommended)'))
    if num_epochs == '1 (Just to test)':
        num_epochs = 1
    if num_epochs == '500 (Recommended)':
        num_epochs = 500
    with st.expander('More info'):
        st.markdown('''With less than 50 epochs the model will not train enough.'\\
            It is recommended to set 500 epochs, the model will stop much earlier if it decides it is done''')

    # Initial learning rate
    initial_learning_rate = st.selectbox('📌 Select the initial learning rate:', 
        ('1e-4', '1e-3', '1e-2 (Recommended)', '1e-1'))
    if initial_learning_rate == '1e-2 (Recommended)':
        initial_learning_rate = 1e-2

    # Choice of the optimizer
    optimizer_choice = st.selectbox('📌 Select the optimizer:', ('Adam (Recommended)', 'Adadelta'))
    if optimizer_choice == 'Adam (Recommended)':
        optimizer_choice = 1
    if optimizer_choice == 'Adadelta':
        optimizer_choice = 2

    # Basemodel
    base_model_choice = st.selectbox(
        '📌 Select the base model:', ('Xception', 'ResNet50V2', 'InceptionV3', 'DenseNet121'))
    if base_model_choice == 'Xception':
        base_model_choice = 1
    if base_model_choice == 'ResNet50V2':
        base_model_choice = 2
    if base_model_choice == 'InceptionV3':
        base_model_choice = 3
    if base_model_choice == 'MobileNetV2':
        base_model_choice = 4
    if base_model_choice == 'DenseNet121':
        base_model_choice = 5

###-----------------------------------------------------------------------------------------
### Code
###-----------------------------------------------------------------------------------------
st.markdown("### **The Code** ✨")

code = f'''
# |--------------------------------------------------------------|
# |                           Settings                           |
# |--------------------------------------------------------------|
# Unzip the dataset
!unzip '/content/drive/MyDrive/dataset.zip'

import os
cwd = os.getcwd()

# Set the dataset path
dataset_dir = cwd + '/dataset'

# Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import cv2
import os

# |--------------------------------------------------------------|
# |                   Train and Validation Set                   |
# |--------------------------------------------------------------|
img_h = {img_h_w}
img_w = {img_h_w}
batch_size = {batch_size}
train_val_split = {train_val_split}

train = ImageDataGenerator(
    rescale = 1/255, 
    validation_split = train_val_split)

train_dataset = train.flow_from_directory(
    dataset_dir, 
    target_size = (img_h, img_w), 
    batch_size  = batch_size, 
    class_mode  = 'categorical',
    subset      = 'training',
    shuffle     = True)

validation_dataset = train.flow_from_directory(
    dataset_dir, 
    target_size = (img_h, img_w), 
    batch_size  = batch_size, 
    class_mode  = 'categorical',
    subset      = 'validation',
    shuffle     = False)

# Define the classes
classes     = train_dataset.class_indices
num_classes = len(classes)

# |--------------------------------------------------------------|
# |                       Model Definition                       |
# |--------------------------------------------------------------|
# Base model definition
base_model_choice = {base_model_choice}

if base_model_choice == 1:  # Xception
    base_model = tf.keras.applications.Xception(
        input_shape = (img_h, img_w, 3), 
        include_top = False, 
        weights     = 'imagenet')

if base_model_choice == 2:  # ResNet50V2
    base_model = tf.keras.applications.ResNet50V2(
        input_shape = (img_h, img_w, 3), 
        include_top = False, 
        weights     = 'imagenet')

if base_model_choice == 3:  # InceptionV3
    base_model = tf.keras.applications.InceptionV3(
        input_shape = (img_h, img_w, 3), 
        include_top = False, 
        weights     = 'imagenet')

if base_model_choice == 4:  # MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape = (img_h, img_w, 3), 
        include_top = False, 
        weights     = 'imagenet')

if base_model_choice == 5:  # DenseNet121
    base_model = tf.keras.applications.DenseNet121(
        input_shape = (img_h, img_w, 3), 
        include_top = False, 
        weights     = 'imagenet')

# Unfreeze all the layers
for layer in base_model.layers:
    layer.trainable = True

# Freeze an initial amount of layers
perc_frozen_layers = {non_trainable}
number_of_layers   = len(base_model.layers)
locked_layers      = math.ceil(perc_frozen_layers*number_of_layers)

for layer in base_model.layers[:locked_layers]:
    layer.trainable = False

# Model definition
inputs = tf.keras.Input(shape = (img_h, img_w, 3))

if base_model_choice == 1:  # Xception
    x = tf.keras.applications.xception.preprocess_input(inputs) 

if base_model_choice == 2:  # ResNet50V2
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs) 

if base_model_choice == 3:  # InceptionV3
    x = tf.keras.applications.inception_v3.preprocess_input(inputs) 

if base_model_choice == 4:  # MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 

if base_model_choice == 5:  # DenseNet121
    x = tf.keras.applications.densenet.preprocess_input(inputs) 

x  = tf.keras.layers.Flatten()(base_model.output)
x  = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model  = tf.keras.models.Model(base_model.input, x)

# |--------------------------------------------------------------|
# |                        Model Training                        |
# |--------------------------------------------------------------|
# Training parameters
epochs = {num_epochs}

# Optimizer
initial_lr = {initial_learning_rate}
optimizer = {optimizer_choice}
if optimizer == 1:
  optimizer = tf.keras.optimizers.Adam(learning_rate = initial_lr)
if optimizer == 2:
  optimizer = tf.keras.optimizers.Adadelta(
      lr      = initial_lr, 
      rho     = 0.95, 
      epsilon = 1e-08, 
      decay   = 0.0)

# Loss Function
loss_function = tf.keras.losses.CategoricalCrossentropy()

# Metrics
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

# Callbacks
filepath = '/content/drive/My Drive/neural_network.hdf5'

Early_Stopping = tf.keras.callbacks.EarlyStopping(
    monitor              = "val_loss", 
    restore_best_weights = False, 
    verbose              = 1, 
    patience             = 10)

LearningRate_Adapter = tf.keras.callbacks.ReduceLROnPlateau(
    monitor  = 'val_loss', 
    factor   = 0.2, 
    patience = 3, 
    verbose  = 1, 
    mode     = 'auto')

Best_model_save = tf.keras.callbacks.ModelCheckpoint(
    filepath       = filepath, 
    save_best_only = True, 
    monitor        = 'val_loss', 
    mode           = 'min', 
    verbose        = True)

callback = [Early_Stopping, LearningRate_Adapter, Best_model_save]

# Train the model
history = model.fit(
    train_dataset, 
    epochs          = epochs, 
    validation_data = validation_dataset, 
    callbacks       = callback)
'''
st.code(code, language='python')

###-----------------------------------------------------------------------------------------
### Predict New Image Label
###-----------------------------------------------------------------------------------------
st.markdown("### Prediction on a new set of images")
st.markdown(''' 
    Now that you have finished training your model you can find out how it performs on new images!
    - Load a folder of new images in the same directory where the model is saved
    - Rename the folder "test"
    - Run the following chunk of code immediately below the previous one
''')

code = '''
# |--------------------------------------------------------------|
# |                     New Image Prediction                     |
# |--------------------------------------------------------------|
# Define class_names
class_names = {}
for (key, value) in classes.items():
  class_names[value] = key

# Load the best model version
model = tf.keras.models.load_model(filepath)

# Load a necessary library
from tensorflow.keras.preprocessing import image

# Define the path
path_test_images = '/content/drive/My Drive/test/'

# For every image in the folder
for i in os.listdir(path_test_images):
  
  # Load image
  img = image.load_img(path_test_images + '/' + i, target_size=(img_h,img_w,3))
  
  # Define the plot
  plt.figure(figsize=(8, 8))
  plt.imshow(img)
  plt.axis("off")

  # Define the label
  X = image.img_to_array(img)
  X = np.expand_dims(X, axis=0)
  images = np.vstack([X])
  for n in range(len(class_names)):
    if np.argmax(model.predict(images))==n:
        plt.title(class_names[n])

'''
st.code(code, language='python')