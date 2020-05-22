import tensorflow  as tf
import numpy as np
import os
import pickle
import time 
import getConfig 
import sys
import random
from cnnModel import cnnModel


gConfig = getConfig.get_config()

def read_data(dataset_path ,im_dim , num_channels ,num_files, images_per_file):
    print('reading')
    files_names = os.listdir(dataset_path)
    dataset_array = np.zeros(shape = (num_files * images_per_file, im_dim, im_dim, num_channels))
    dataset_labels = np.zeros(shape = (num_files*images_per_file),dtype = np.uint8)
    index = 0 
    for file_name in files_names:

        if file_name[:len(file_name)-1] == 'data_batch_':
            print('dealing with :',file_name)
            data_dict = unpickle_patch(dataset_path + file_name)
            images_data = data_dict[b'data']
            print(images_data.shape)
            images_data_reshaped = np.reshape(
                    images_data,newshape =(len(images_data),im_dim,im_dim,num_channels))
            dataset_array[index*images_per_file:(index+1)*images_per_file,:,:,:]=images_data_reshaped
            dataset_labels[index*images_per_file:(index+1)*images_per_file] = data_dict[b'labels']

            return dataset_array,dataset_labels

def unpickle_patch(file):
    patch_bin_file = open(file,'rb')
    patch_dict = pickle.load(patch_bin_file ,encoding = 'bytes')
    return patch_dict

def create_model():
    if 'pretrained_model' in gConfig:
        model  = tf.keras.models.load_model(gConfig['pretrained_model'])
        return model

    ckpt =tf.io.gfile.listdir(gConfig['working_directory'])
    
    if ckpt:
        model_file = os.path.join(gConfig['working_directory'],ckpt[-1])
        print('reading model parameters from %s'%model_file)
        model = tf.keras.models.load_model(model_file)
        return model

    else:
        model=cnnModel( gConfig['rate'])
        model =model.create_model()
        return model

dataset_array,dataset_labels=read_data(
        dataset_path = gConfig['dataset_path'],
        im_dim=gConfig['im_dim'],
        num_channels =gConfig['num_channels'],
        num_files = gConfig['num_files'],
        images_per_file = gConfig['images_per_file'])

dataset_array = dataset_array.astype('float32')/255

dataset_labels = tf.keras.utils.to_categorical(dataset_labels,10)

def train():
    model = create_model()
    print('X shape:',dataset_array.shape)
    print('Y shape:',dataset_labels.shape)
    history = model.fit(dataset_array,dataset_labels,verbose=1,epochs=100,validation_split=.2)
    filename = 'cnn_model.h5'
    checkpoint_path = os.path.join(gConfig['working_directory'],filename)
    model.save(checkpoint_path)

def predict(data):
    ckpt = os.listdir(gConfig['working_directory'])
    checkpoint_path = os.path.join(gConfig['working_dorectory'],'cnn_model.h5')
    model = tf.keras.models.load_model(checkpoint_path)
    prediction = model.predict(data)

    index = tf.math.argmax(prediction[0]).numpy()
    return label_name_dict[index]


if __name__=='__main__':
    gConfig = getConfig.get_config()
    if gConfig['mode']=='train':
        train()
    elif gConfig['mode'] =='predict':
        print('use python3 app.py')




