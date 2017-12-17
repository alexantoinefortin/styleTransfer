import os, imp, numpy as np
t = imp.load_source('tools', './tools.py')
bs = imp.load_source('IncreaseBSOnPlateau', './IncreaseBSOnPlateau.py')

path_to_save_models = '../models/mnist-simple'
#======================
#Loading data in memory
#======================
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and reshape into a vector of size 784
x_train, x_test = t.preprocess_input(x_train), t.preprocess_input(x_test)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print(x_train.shape)
#==================
#Model architecture
#==================
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.regularizers import l1
from keras.optimizers import SGD
from keras.models import Model

# this is our input placeholder
input_img = Input(shape=(28, 28, 1))
# "encoded" is the encoded representation of the input
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x) #the representation is (4, 4, 8)
# "decoded" is the lossy reconstruction of the input
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(4,4,8,)) #XXX: hard-coded value
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
LR = 0.1
MIN_LR = 0.0001
BATCH_SIZE = 128
MAX_BATCH_SIZE = 4500
INCREASE_RATE = 3
for i in range(100):
    if i==0:
        new_bs, new_lr = BATCH_SIZE, LR
        sgd = SGD(lr=new_lr, decay=0, momentum=0.9, nesterov=True)
        autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
        calc_new_bs_and_lr = bs.IncreaseBSOnPlateau(model=autoencoder,
                     monitor='val_loss', factor_bs=INCREASE_RATE, patience=0,
                     verbose=1, mode='auto', epsilon=1e-4, cooldown=0,
                     max_bs=6000, min_lr=MIN_LR)
        calc_new_bs_and_lr.on_train_begin()
    else:
        calc_new_bs_and_lr.update_model(autoencoder)
        new_bs, new_lr = calc_new_bs_and_lr.on_epoch_end(epoch = i+1)
        sgd = SGD(lr=new_lr, decay=0, momentum=0.9, nesterov=True)
        autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
    print("Epoch:{}, BATCH_SIZE: {}, LR:{}".format(i, new_bs, new_lr))
    autoencoder.fit(x_train, x_train,
                epochs=i+1,
                batch_size=new_bs,
                shuffle=True,
                validation_data=(x_test, x_test),
                initial_epoch=i)
                #XXX: get autoencoder.val_loss and <> BATCH_SIZE On Plateau

print("It took {}sec to train the model".format(time()-startTime))
print(autoencoder.history.history)
print("Params:{}, batch_size:{}".format(autoencoder.history.params, autoencoder.history.params.get('batch_size','')))
