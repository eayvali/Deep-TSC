# -*- coding: utf-8 -*-
"""
Created on Tue May 19 00:25:26 2020

@author: elif.ayvali
Implementation of :
Karim, Fazle, et al. "Multivariate lstm-fcns for time series classification." 
Neural Networks 116 (2019): 237-245.
Model from https://github.com/titu1994/MLSTM-FCN
"""
from utils import calculate_metrics,save_test_duration
import numpy as np
import time
import keras
import os
from attention_LSTM_layer import AttentionLSTM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import class_weight



class Parallel_CNNLSTM: 
    
    def __init__(self,data_file, output_folder):
        """        
        Parameters
        ----------
        data_file : .npz training data 
        output_folder : where to save model output and results
        """
        #Initialize parameters
        self.callbacks = None
        self.nb_epochs = 200
        self.attention=True
        self.n_feature_maps=32
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
    
    def build_model(self):
        '''
        If n_samples are not uniform use Masking layer after input before LSTM
        Trick: to speed up LSTM, shuffle input as (n_features,n_samples) 
               it has no significant effect on accuracy   
        Permute:Tuple of integers. Permutation pattern, does not include
                 the samples dimension. Indexing starts at 1.
        Input to CNN is n_samples,n_features(channels)
        '''
        input_shape= self.X_train.shape[1:]# (n_samples x n_features)
        
        input_layer = keras.layers.Input(input_shape) 
            
        #LSTM branch
        x=keras.layers.Permute((2, 1))(input_layer)
        if self.attention is True:
            x =AttentionLSTM(50)(x)            
        else:
            x =keras.layers.LSTM(50)(x)
        lstm_head =keras.layers. Dropout(0.2)(x)
        #CNN branch
        y = keras.layers.Conv1D(self.n_feature_maps, 8, padding='same', kernel_initializer='he_uniform')(input_layer)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        y = self.__squeeze_excite_block(y)
    
        y = keras.layers.Conv1D(self.n_feature_maps, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        y = self.__squeeze_excite_block(y)
    
        y = keras.layers.Conv1D(self.n_feature_maps, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        cnn_head = keras.layers.GlobalAveragePooling1D()(y)
    
        x = keras.layers.concatenate([lstm_head, cnn_head])

        output_layer = keras.layers.Dense(self.n_classes, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_folder + '/best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

       #pydot and  graphviz (for Windows) needs to be installed for next line
        try:
            keras.utils.plot_model(model,to_file=self.output_folder + '/Parallel_CNN_LSTM.png',show_shapes=False,show_layer_names=True )
        except:
            print('cannot save the model architecture') 
            
        return model
        
    def __squeeze_excite_block(self,input_tensor):
        ''' Create a squeeze-excite block
        Args:
            input_tensor: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''
        filters = input_tensor._keras_shape[-1] # channel_axis = -1 for TF
    
        se = keras.layers.GlobalAveragePooling1D()(input_tensor)
        se = keras.layers.Reshape((1, filters))(se)
        se = keras.layers.Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = keras.layers.multiply([input_tensor, se])
        return se


    def fit(self, plot_test_acc=False):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        x_val=self.X_test
        y_val=self.y_test
        if self.batch_size is None:
            mini_batch_size = int(min(self.X_train.shape[0] / 10, 64))
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
        model = keras.models.load_model(model_path, custom_objects = {'AttentionLSTM': AttentionLSTM})
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
        self.fit(plot_test_acc=False)      

    def test_trained_model(self, output_folder):
        model_file = output_folder + '/best_model.hdf5'
        model = keras.models.load_model(model_file, custom_objects = {'AttentionLSTM': AttentionLSTM})
        batch_size = 1
        y_pred = model.predict(self.X_test, batch_size = batch_size)
        y_pred = np.argmax(y_pred, axis = 1)#get integer labels
        metrics = calculate_metrics(self.y_true, y_pred, 0.0,self.output_folder)
        return metrics  
    
# print('...Training (Many To One) CNNLSTM Classifier (Parallel Architecture) \n')
# clf=Parallel_CNNLSTM(data_file='../data/UCI_HAR.npz', output_folder='../results/Parallel_CNNLSTM')
# clf.run()


clf=Parallel_CNNLSTM(data_file='../data/UCI_HAR.npz', output_folder='../results/Parallel_CNNLSTM')
clf.test_trained_model(output_folder='../results/Parallel_CNNLSTM')

