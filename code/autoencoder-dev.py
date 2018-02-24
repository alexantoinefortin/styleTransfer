#!/usr/local/bin/python3.6
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.65
session = tf.Session(config=config)

import imp, numpy as np
from time import time
t = imp.load_source('tools', './code/tools.py')
# Since we are using MaxPooling2D/UpSampling2D, the input shape must be
# divisible by 2^5 (there are 5 MaxPooling2D operations.). 128, 256, 512, ...
TARGET_SIZE=(256, 256, 3)
BATCH_SIZE = 100
# Image generators
startTime = time()
genTrain = t.genTrain()
genTest = t.genTest()
traingen = t.traingen(path='./inputs/train2017', genTrain=genTrain, BATCH_SIZE=BATCH_SIZE, RANDOM_CROP_SIZE=TARGET_SIZE)
testgen = t.testgen(path='./inputs/val2017', genTest=genTest, BATCH_SIZE=BATCH_SIZE, RANDOM_CROP_SIZE=TARGET_SIZE)
print("The image generators took {}s to initiate.".format(int(time()-startTime)))

#=================
# Model Parameters
#=================
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
LEVEL = 1 # 1,2,3,4,5: train them turn-by-turn in this order
MODEL_NAME = 'level{}_decoder'.format(LEVEL)
NB_EPOCH = 25
NB_STEPS_PER_EPOCH = int(118287/BATCH_SIZE) # Number of images in train set
NB_VAL_STEPS_PER_EPOCH = int(5000/BATCH_SIZE)
INIT_LR = 0.1
REDUCE_LR_ON_PLATEAU_FACTOR = 0.5 # every plateau, multiply by this number

# Define model
with tf.device('/cpu:0'):
    e, a = t.define_model(target_size=TARGET_SIZE, level=1) # encoder, autoencoder
    # quick check
    for layer in a.layers:
        print("name:{:18s}\ttrainable:{}, output_shape:{}".format(layer.name, layer.trainable, layer.output_shape))
    optimizer = SGD(lr=INIT_LR, decay=0, momentum=0.9, nesterov=True)
    # Callbacks
    cp = t.cp(path_to_save='./models/{}'.format(MODEL_NAME), save_weights_only=True)
    r_lr = ReduceLROnPlateau(   monitor='val_loss',
                                factor=REDUCE_LR_ON_PLATEAU_FACTOR,
                                patience=1,
                                verbose=10,
                                mode='min',
                                epsilon=0.0001,
                                cooldown=0,
                                min_lr=0.0001)
    # Custom loss
    def custom_loss(y_true, y_pred):
        pixel_loss = K.mean(K.square(y_pred - y_true), axis=-1)
        # Feature loss: To do this, we pass y_true and y_pred into our fixed encoder e.
        # Construct a graph to evaluate feature_pred and feature_true
        feature_pred, feature_true = y_pred, y_true
        for layer in e.layers:
            feature_pred = layer(feature_pred)
            feature_true = layer(feature_true)
        feature_loss = K.mean(K.square(feature_pred-feature_true), axis=-1)
        return pixel_loss + feature_loss

#=======================================================
# compile the model
# should be done *after* setting layers to non-trainable
#=======================================================
parallel_model = multi_gpu_model(a, gpus=2)
parallel_model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae', 'mse'])

print("NB_EPOCH:{}".format(NB_EPOCH))
print("nb of minibatches to process (train):{}".format(NB_STEPS_PER_EPOCH))
print("nb of minibatches to process (test) :{}".format(NB_VAL_STEPS_PER_EPOCH))
parallel_model.fit_generator(
                            generator=traingen,
                            steps_per_epoch=NB_STEPS_PER_EPOCH,
                            epochs=NB_EPOCH,
                            use_multiprocessing=False,
                            max_queue_size=30,
                            workers=20,
                            callbacks=[r_lr],
                            validation_data=testgen,
                            validation_steps=NB_VAL_STEPS_PER_EPOCH)

a.save_weights('./models/{}/level{}_end_model.hdf5'.format(MODEL_NAME, LEVEL))
