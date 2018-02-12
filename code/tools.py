#!/usr/local/bin/python3
"""
Alex-Antoine Fortin
Wednesday Dec 6, 2017
Description
This python script regroups functions used thorough our code
challenge
"""
import os, imp, collections, threading, numpy as np
from numba import jit
#from keras.preprocessing.image import ImageDataGenerator
image = imp.load_source('image', './code/ImageGenerator.py')
from keras.callbacks import ModelCheckpoint
#===============
#Cropping images
#===============
@jit
def preprocess_input(x):
    x = x.astype('float32') / 255.
    return x

def postprocess_int(x):
    x *= 255.
    return x

def center_crop(x, center_crop_size, **kwargs):
    """
    Returns a preprocessed centered crop of an array x
    """
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return preprocess_input(x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh])

def random_crop(x, random_crop_size, sync_seed=None):
    """
    Return a preprocessed random crop of an array x
    @random_crop_size: Tuple. (width, height, channels)
    """
    np.random.seed(sync_seed)
    w, h = x.shape[0], x.shape[1]
    rangew = (w - random_crop_size[0])
    rangeh = (h - random_crop_size[1])
    #cropped = np.empty(shape=(random_crop_size[0], random_crop_size[1], random_crop_size[2]), dtype=float) #initialize empty array
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped = x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1],:]
    return cropped

#==================
# Defining data generator
#==================
def genTrain():
    return image.ImageDataGenerator(
    featurewise_center=False, # Set input mean to 0 over the dataset, feature-wise.
    samplewise_center=False, # Set each sample mean to 0.
    featurewise_std_normalization=False, # Divide inputs by std of the dataset, feature-wise
    samplewise_std_normalization=False, # Divide each input by its std
    zca_whitening=False, # Apply ZCA whitening.
    zca_epsilon=1e-6, # epsilon for ZCA whitening. Default is 1e-6
    rotation_range=0, # Int. Degree range for random rotations
    width_shift_range=0.05, # Float (fraction of total width). Range for random horizontal shifts.
    height_shift_range=0.05, # Float (fraction of total height). Range for random vertical shifts.
    shear_range=0., # Float. Shear Intensity (Shear angle in counter-clockwise direction as radian
    zoom_range=0., # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
    channel_shift_range=0., # Float. Range for random channel shifts.
    fill_mode='reflect', #  {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode
    cval=0., # Value used for points outside the boundaries when fill_mode = "constant"
    horizontal_flip=True, # Randomly flip inputs horizontally.
    vertical_flip=False, # Randomly flip inputs vertically.
    rescale=None, # If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation)
    preprocessing_function=preprocess_input, # The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.
    data_format='channels_last')

def genTest():
    """
    For testing, we only flip and normalize the image
    """
    return image.ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    data_format='channels_last'
    )

def traingen(path, genTrain, BATCH_SIZE, RANDOM_CROP_SIZE):
    return genTrain.flow_from_directory(
        directory=path,
        target_size= None, # default: (256, 256)
        color_mode="rgb", # one of "grayscale", "rbg"
        classes=None, # If not provided, does not return a label.
        batch_size=BATCH_SIZE, # default: 32.
        shuffle=True,
        #seed=1234,
        #save_to_dir='../generated_img', #Path to directory where to save generated pics
        class_mode='input', #one of "categorical", "binary", "sparse" or None
        random_crop=random_crop,
        random_crop_size=RANDOM_CROP_SIZE
        )

def testgen(path, genTest, BATCH_SIZE, RANDOM_CROP_SIZE):
    return genTest.flow_from_directory(
        directory=path,
        target_size= None,
        batch_size=BATCH_SIZE,
        class_mode='input',
        random_crop=random_crop,
        random_crop_size=RANDOM_CROP_SIZE)

def holdoutgen(genTest, BATCH_SIZE, path='../input/holdout'):
    return genTest.flow_from_directory(
        path,
        target_size= (180, 180),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

#=======================================
# Adding random cropping to a generators
#=======================================
class crop_gen:
    """
    Uses @generator to get 1 batch.
    crop_gen randomly select a crop of size @target_size from an image obtained
    via @generator. It then return the randomly cropped image and its label if
    @with_label.
    """
    def __init__(self, generator,target_size):
        self.lock = threading.Lock()
        self.generator = generator
        self.target_size = target_size
    #
    def __iter__(self):
        return self
    #
    def __next__(self):
        with self.lock:
            batch = next(self.generator)
        return (random_crop(batch[0], self.target_size),batch[1])

def cp(path_to_save, save_weights_only=False):
    """
    Defines model checkpoint
    """
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
    return ModelCheckpoint(filepath=os.path.join(path_to_save,
                        'model-{epoch:03d}-{val_loss:.4f}.hdf5'),
                        monitor='mse',
                        mode='min', # min: for loss, max: for acc, auto: infer from monitor's value
                        save_best_only=False, # if True, the latest best model according to the quantity monitored will not be overwritten
                        save_weights_only=save_weights_only, # model.save_weights(filepath) if True else model.save(filepath)
                        period=2 # Perform ModelCheckpoint every period epoch
                        )

#=================
# Model generators
#=================
def define_model(target_size, level):
    """
    @target_size: Tuple. Size of the input image (width, height, channels)
    eg: (256, 256, 3)
    @level: Int. level of encoding. Load the first level block of convolution
    from VGG19-imageNet.
    """
    from keras.applications.vgg19 import VGG19
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
    from keras.models import Model, model_from_yaml
    from keras import backend as K
    from keras.initializers import VarianceScaling
    MSRA = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None)
    # level_dict is a dictionary of namedtuples with actions to take depending on the value of level
    level_def = collections.namedtuple('level_def', 'pop_from_encoder load_decoders')
    level_dict ={
    1:level_def(pop_from_encoder=20, load_decoders=None),
    2:level_def(pop_from_encoder=17, load_decoders='level1_decoder'),
    3:level_def(pop_from_encoder=14, load_decoders='level2_decoder'),
    4:level_def(pop_from_encoder=9, load_decoders='level3_decoder'),
    5:level_def(pop_from_encoder=4, load_decoders='level4_decoder')
                }

    encoder = VGG19(include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(target_size[0], target_size[1], target_size[2]),
    pooling=None, classes=None)
    # Freeze encoder
    for layer in encoder.layers:
        layer.trainable = False
    # Remove undesired layers from the encoder
    for _ in range(level_dict.get(level).pop_from_encoder):
        encoder.layers.pop()
    encoder.outputs = [encoder.layers[-1].output]
    encoder.layers[-1].outbound_nodes = []

    # Setting-up decoder depending on level
    if level==5:
        decoder = Conv2D(512, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block5_conv4')(encoder.outputs[-1])
        decoder = UpSampling2D((2, 2), name='de_block5_pool2')(decoder)
        decoder = Conv2D(512, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block5_conv3')(decoder)
        decoder = Conv2D(512, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block5_conv2')(decoder)
        decoder = Conv2D(512, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block5_conv1')(decoder)
    elif level==4:
        decoder = Conv2D(256, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block4_conv4')(encoder.outputs[-1])
        decoder = UpSampling2D((2, 2), name='de_block4_pool')(decoder)
        decoder = Conv2D(256, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block4_conv3')(decoder)
        decoder = Conv2D(256, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block4_conv2')(decoder)
        decoder = Conv2D(256, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block4_conv1')(decoder)
    elif level==3:
        decoder = Conv2D(128, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block3_conv2')(encoder.outputs[-1])
        decoder = UpSampling2D((2, 2), name='de_block3_pool')(decoder)
        decoder = Conv2D(128, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block3_conv1')(decoder)
    elif level==2:
        decoder = Conv2D(64, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block2_conv1')(encoder.outputs[-1])
        decoder = UpSampling2D((2, 2), name='de_block2_pool')(decoder)
    elif level==1:
        decoder = Conv2D(64, (3, 3), activation='relu', kernel_initializer=MSRA, padding='same', name='de_block1_conv1')(encoder.outputs[-1])
        decoder = Conv2D(3, (3, 3), activation='linear', kernel_initializer=MSRA, padding='same', name='output_block')(decoder)
    # decoder_top is the pretrained decoder for finer-level decoder
    if type(level_dict.get(level).load_decoders)!=type(None): # Do nothing if level==1
        tmp_top = model_from_yaml(level_dict.get(level).load_decoders+'.yaml')
        tmp_top = tmp_top.load_weights(level_dict.get(level).load_decoders+'.hdf5', by_name=False)
        for layer in tmp_top.layers: # Do not train top decoder
            layer.trainable = False
        decoder_top = Model(input=[decoder], output=[tmp_top.output])
    else: # if training level==1
        decoder_top = decoder
    # Putting the model pieces together
    autoencoder = Model(input=[encoder.input], output=[decoder_top])
    return encoder, autoencoder
