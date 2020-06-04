# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:11:21 2020

@author: elif.ayvali

Implementation of:
Fawaz, Hassan Ismail, et al. 
"InceptionTime: Finding AlexNet for Time Series Classification."
arXiv preprint arXiv:1909.04939 (2019).
Model from https://github.com/hfawaz/InceptionTime
"""
from utils import calculate_metrics,save_test_duration
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import class_weight


import numpy as np
import time
import keras
import os

class InceptionTime:
    
    def __init__(self, data_file, output_folder):
        """        
        Parameters
        ----------
        data_file : .npz training data 
        output_folder : where to save model output and results
        """

        #Initialize parameters                                                                           
        self.nb_filters =32
        self.use_residual = True
        self.use_bottleneck = True
        self.depth = 6
        self.kernel_size = 41 - 1
        self.callbacks = None
        self.bottleneck_size = 32
        self.nb_epochs = 200
        self.verbose=True        
        self.use_class_weight=True
        
        
        if not os.path.exists(output_folder):
           os.makedirs(output_folder) 
           
        dataOriginal = np.load(data_file)
        data = {}
        data['features'] = dataOriginal['features']
        data['labels'] = dataOriginal['labels']                
        

        self.output_folder = output_folder
        self.n_windows, self.n_samples, self.n_features, = data['features'].shape
        self.n_classes=len(np.unique(data['labels']))
        
        #Structure input data shape: X_train (N,T,F):(windows,sample_per_window,feature) , y_train one hot encoded
        self.X_train, self.X_test, y_train, y_test = self.__split_data( data['features'],data['labels'] ,0.3)
        # save original y for accuracy calculation
        self.y_true = y_test
        self.y_true_train = y_train        
        
        #Convert integer labels to one hot labels for classification 
        enc = preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        self.y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        self.y_test = enc.transform(y_test.reshape(-1, 1)).toarray()                                                                    
                                                                     

        self.batch_size = int(min(self.X_train.shape[0] / 10, 16))
 
        self.model = self.build_model()
        if (self.verbose == True):
            self.model.summary()
        self.model.save_weights(self.output_folder + '/model_init.hdf5')
        
    def __split_data(self,data,labels,split_ratio):
        #Divide training and test data: 
        X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=split_ratio,shuffle=False) #preserves indeces  
        y_train=y_train.reshape(-1).astype(np.int64)
        y_test=y_test.reshape(-1).astype(np.int64)        
        
        return X_train, X_test, y_train, y_test
        
       
    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))
        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x
    
    def build_model(self):
        input_shape= self.X_train.shape[1:]# (n_samples , n_features)
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(self.n_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_folder + '/best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]
       #pydot and  graphviz (for Windows) needs to be installed for next line
        try:
            keras.utils.plot_model(model,to_file=self.output_folder + '/InceptionTime.png',show_shapes=False,show_layer_names=True )
        except:
            print('cannot save the model architecture') 

        return model
        
 
    def fit(self, plot_test_acc=False):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        x_val=self.X_test
        y_val=self.y_test
        if self.batch_size is None:
            mini_batch_size = int(min(self.X_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size
            
            
            
        if self.use_class_weight is True:
            w_class = class_weight.compute_class_weight('balanced',np.unique(self.y_true_train),self.y_true_train)
        else:
            w_class=np.ones(self.n_classes)    
            
            
        start_time = time.time()

        if plot_test_acc:

            hist = self.model.fit(self.X_train, self.y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,class_weight=w_class,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:

            hist = self.model.fit(self.X_train, self.y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,class_weight=w_class,
                                  verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_folder + '/last_model.hdf5')

        y_pred = self.__predict(return_df_metrics=False)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)
        # save predictions
        np.save(self.output_folder + '/y_pred.npy', y_pred)


        keras.backend.clear_session()

    
    def __predict(self,  return_df_metrics=True):
        
        start_time = time.time()
        model_path = self.output_folder + '/best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(self.X_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            metrics = calculate_metrics(self.y_true, y_pred, 0.0,self.output_folder)
            return metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_folder + '/test_duration.csv', test_duration)
            return y_pred
        
        
        
    def run(self):
    #you can call multiple iterations here to find mean performance
        self.fit(plot_test_acc=False)  
    
    def test_trained_model(self, output_folder):
        model_file = output_folder + '/best_model.hdf5'
        model = keras.models.load_model(model_file)
        batch_size = 1
        y_pred = model.predict(self.X_test, batch_size = batch_size)
        y_pred = np.argmax(y_pred, axis = 1)#get integer labels
        metrics = calculate_metrics(self.y_true, y_pred, 0.0,self.output_folder)
        return metrics      

        
print('...Training (Many To One) Inception-Time Classifier \n')
clf=InceptionTime(data_file='../data/UCI_HAR.npz', output_folder='../results/InceptionTime')
clf.run()
# clf=InceptionTime(data_file='../data/UCI_HAR.npz', output_folder='../results/InceptionTime')
# clf.test_trained_model(output_folder='../results/InceptionTime')

