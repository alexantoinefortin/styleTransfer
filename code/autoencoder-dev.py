#!/usr/local/bin/python3.6
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import imp
from time import time
t = imp.load_source('tools', './code/tools.py')
# Since we are using MaxPooling2D/UpSampling2D, the input shape must be
# divisible by 2^5 (there are 5 MaxPooling2D operations.). 128, 256, 512, ...
TARGET_SIZE=(128, 128, 3) # XXX: Some COCO images are smaller than 256x256 (should I remove them from train&testing sets?)
BATCH_SIZE = 128
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
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
LEVEL = 1 # 1,2,3,4,5: train them turn-by-turn in this order
MODEL_NAME = 'level{}_decoder'.format(LEVEL)
NB_EPOCH = 65
NB_STEPS_PER_EPOCH = int(118287/BATCH_SIZE) # Number of images in train set
NB_VAL_STEPS_PER_EPOCH = int(5000/BATCH_SIZE)
INIT_LR = 0.1
REDUCE_LR_ON_PLATEAU_FACTOR = 0.5 # every plateau, multiply by this number

# Define model
e, a = t.define_model(target_size=TARGET_SIZE, level=1)
# quick check
for layer in a.layers:
    print("name:{:18s}\ttrainable:{}, output_shape:{}".format(layer.name, layer.trainable, layer.output_shape))

optimizer = SGD(lr=INIT_LR, decay=0, momentum=0.9, nesterov=True)
# Callbacks
cp = t.cp(path_to_save='./models/{}'.format(MODEL_NAME), save_weights_only=True)
r_lr = ReduceLROnPlateau(   monitor='val_loss',
                            factor=REDUCE_LR_ON_PLATEAU_FACTOR,
                            patience=1,
                            verbose=0,
                            mode='auto',
                            epsilon=0.0001,
                            cooldown=0,
                            min_lr=0.0001)
#=======================================================
# compile the model
# should be done *after* setting layers to non-trainable
#=======================================================
a.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

print("NB_EPOCH:{}".format(NB_EPOCH))
print("nb of minibatches to process (train):{}".format(NB_STEPS_PER_EPOCH))
print("nb of minibatches to process (test) :{}".format(NB_VAL_STEPS_PER_EPOCH))
a.fit_generator(
        generator=traingen,
        steps_per_epoch=NB_STEPS_PER_EPOCH,
        epochs=NB_EPOCH,
        workers=12,
        callbacks=[cp, r_lr],
        validation_data=testgen,
        validation_steps=NB_VAL_STEPS_PER_EPOCH)
