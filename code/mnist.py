import os, imp, numpy as np
t = imp.load_source('tools', './tools.py')

path_to_save_models = '../models/mnist-simple'
#======================
#Loading data in memory
#======================
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and reshape into a vector of size 784
x_train, x_test = t.preprocess_input(x_train), t.preprocess_input(x_test)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


#==================
#Model architecture
#==================
from keras.layers import Input, Dense
from keras.regularizers import l1
from keras.optimizers import SGD
from keras.models import Model

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(128, activity_regularizer=l1(10e-5),
                activation='relu', name='ENCODER_1')(input_img)
encoded = Dense(64, activity_regularizer=l1(10e-5),
                activation='relu', name='ENCODER_2')(encoded)
encoded = Dense(32, activity_regularizer=l1(10e-5),
                activation='relu', name='BOTTLENECK_1')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activity_regularizer=l1(10e-5),
                activation='relu', name='DECODER_1')(encoded)
decoded = Dense(128, activity_regularizer=l1(10e-5),
                activation='relu', name='DECODER_2')(decoded)
decoded = Dense(784, activation='sigmoid', name='OUTPUT')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(32,)) #XXX: hard-coded value
# retrieve the last layer of the autoencoder model
decoder_layer_1 = autoencoder.layers[-3]
decoder_layer_2 = autoencoder.layers[-2]
decoder_layer_3 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer_3(decoder_layer_2(decoder_layer_1(encoded_input))))

#=========
#Callbacks
#=========
if not os.path.isdir(path_to_save_models):
    os.makedirs(path_to_save_models)
cp = t.cp(path_to_save_models)


#==========
#Train Hard
#==========
from time import time
startTime = time()
LR = 0.5
MIN_LR = 0.0001
BATCH_SIZE = 640
MAX_BATCH_SIZE = 6000
INCREASE_RATE = 1.5
for i in range(100):
    if i==0:
        BATCH_SIZE=BATCH_SIZE
        sgd = SGD(lr=LR, decay=0, momentum=0.9, nesterov=True)
        autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
    elif i%10==0:
        tmp_BATCH_SIZE = int(min(BATCH_SIZE*INCREASE_RATE, MAX_BATCH_SIZE))
        if BATCH_SIZE == tmp_BATCH_SIZE: # Reached MAX_BATCH_SIZE
            LR = max(LR/INCREASE_RATE, MIN_LR)
            sgd = SGD(lr=LR, decay=0, momentum=0.9, nesterov=True)
            autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
        BATCH_SIZE = tmp_BATCH_SIZE
    print("Epoch:{}, BATCH_SIZE: {}, LR:{}".format(i, BATCH_SIZE, LR))
    autoencoder.fit(x_train, x_train,
                epochs=i+1,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                initial_epoch=i)
                #XXX: get autoencoder.val_loss and <> BATCH_SIZE On Plateau

print("It took {}sec to train the model".format(time()-startTime))
