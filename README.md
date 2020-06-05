# Deep-TSC: Multivariate Time Series Classification

This repository shows the use of three different TSC algorithms on the [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

## Dateset

- Data was collected using a cellphone attached to the waists of test subjects.  Each person performs 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) 
- Data consists of  3-axial linear acceleration and 3-axial angular velocity  collected at a constant rate of 50Hz using  the embedded accelerometer and gyroscope of the phone.
- To process training data from raw data , follow instructions [here](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/)

![HAR-class-dist](./pics/HAR-class-dist.png)

## Implementations
InceptionTime [2] uses 1D convolutions similar to CNNs used for image classification.
![models](./pics/models1.png)

Parallel CNN-LSTM uses CNN layers and an attention LSTM layer in parallel [3]. This is in contrast to common serial CNN LSTM architectures, where CNNs are used to encode features and the LSTM layer is used to model long-term dependencies of the features across time. The authors in [3] shuffle operation before the attention LSTM layer for computational efficiency. Global temporal information of each feature is fed ti LSTM at once.
Serial CNN LSTM uses temporal convolutional networks as the CNN layers, which incorporate dilated causal convolutions to increase the receptive field of the CNN layers[4]. 



![models](./pics/models2.png)

> Note: Using class_weights changes the range of the loss. This may affect the stability of the training depending on the optimizer. Optimizers whose step size is dependent on the magnitude of the gradient, like optimizers.SGD, may fail. The optimizer used here, optimizers.Adam, is unaffected by the scaling change. Also note that because of the weighting, the total losses are not comparable between the two models.

## Files

You can directly run the scripts below:

**/models**

- Parallel_CNNLSTM_SA.py
- Serial_CNNLSTM_SA.py
- InceptionTime_SA.py

**/data**

- UCI_HAR.npz
   -data['features']: number of windows,samples per window, number of features
   -data['labels'] :  class label per window


## Dependencies

Tested on ubuntu 18.04
* conda 4.8.3
* Keras 2.3.1
* tensorflow 1.14.0
* pydot 1.4.1


## References

Code references were cited in the scripts.

_[1]_  Anguita, Davide, et al. "A public domain dataset for human activity recognition using smartphones." Esann. 2013.

_[2]_ Fawaz, Hassan Ismail, et al. "InceptionTime: Finding AlexNet for Time Series Classification." arXiv preprint 
arXiv:1909.04939 (2019).

_[3]_ Karim, Fazle, et al. "Multivariate lstm-fcns for time series classification." Neural Networks 116 (2019): 237-245.

_[4]_ Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio." arXiv preprint arXiv:1609.03499 (2016).
