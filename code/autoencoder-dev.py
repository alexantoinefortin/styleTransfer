import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import collections
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, model_from_yaml
from keras import backend as K
target_size=(256, 256, 3)
# Since we are using MaxPooling2D/UpSampling2D, the input shape must be
# divisible by 2^5 (there are 5 MaxPooling2D operations.). 128, 256, 512, ...

level_def = collections.namedtuple('level_def', 'pop_from_encoder load_decoders')
level_dict ={
1:level_def(pop_from_encoder=18, load_decoders=[]),
2:level_def(pop_from_encoder=15, load_decoders=['level1_decoder']),
3:level_def(pop_from_encoder=10, load_decoders=['level2_decoder', 'level1_decoder']),
4:level_def(pop_from_encoder=5, load_decoders=['level3_decoder', 'level2_decoder', 'level1_decoder']),
5:level_def(pop_from_encoder=0, load_decoders=['level4_decoder', 'level3_decoder', 'level2_decoder', 'level1_decoder'])
            }

def define_model(target_size, level):
    """
    @target_size: Tuple. Size of the input image (width, height, channels)
    eg: (256, 256, 3)
    @level: Int. level of encoding. Load the first level block of convolution
    from VGG19-imageNet.
    """
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
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block5_conv1')(encoder.output)
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block5_conv2')(decoder)
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block5_conv3')(decoder)
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block5_conv4')(decoder)
        decoder = UpSampling2D((2, 2), name='de_block5_pool2')(decoder) #XXX: UpSampling2D with nearest neibs?
    elif level==4:
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block4_conv1')(encoder.output)
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block4_conv2')(decoder)
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block4_conv3')(decoder)
        decoder = Conv2D(512, (3, 3), activation='relu', padding='same', name='de_block4_conv4')(decoder)
        decoder = UpSampling2D((2, 2), name='de_block4_pool')(decoder)
    elif level==3:
        decoder = Conv2D(256, (3, 3), activation='relu', padding='same', name='de_block3_conv1')(encoder.output)
        decoder = Conv2D(256, (3, 3), activation='relu', padding='same', name='de_block3_conv2')(decoder)
        decoder = Conv2D(256, (3, 3), activation='relu', padding='same', name='de_block3_conv3')(decoder)
        decoder = Conv2D(256, (3, 3), activation='relu', padding='same', name='de_block3_conv4')(decoder)
        decoder = UpSampling2D((2, 2), name='de_block3_pool')(decoder)
    elif level==2:
        decoder = Conv2D(128, (3, 3), activation='relu', padding='same', name='de_block2_conv1')(encoder.output)
        decoder = Conv2D(128, (3, 3), activation='relu', padding='same', name='de_block2_conv2')(decoder)
        decoder = UpSampling2D((2, 2), name='de_block2_pool')(decoder)
    elif level==1:
        decoder = Conv2D(64, (3, 3), activation='relu', padding='same', name='de_block1_conv1')(encoder.output)
        decoder = Conv2D(64, (3, 3), activation='relu', padding='same', name='de_block1_conv2')(decoder)
        decoder = UpSampling2D((2, 2), name='de_block1_pool')(decoder)
        decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='output_block')(decoder)
    # decoder_top is the pretrained decoder for finer-level decoder
    for idx, tops in enumerate(level_dict.get(level).load_decoders):
        tmp_top = model_from_yaml(tops+'.yaml')
        tmp_top = tmp_top.load_weights(tops+.'hdf5', by_name=False)
        if idx==0: #First iteration of the for-loop
            tmp_decoder_top = tmp_top
        else:
            tmp_decoder_top = Model(input=[tmp_decoder_top.output], output=[tmp_top.output])
    if len(level_dict.get(level).load_decoders): # Do nothing if level==1
        decoder_top = tmp_decoder_top
        for layer in decoder_top.layers: # Do not train top decoder
            layer.trainable = False
    # TODO: Putting the model pieces together
        decoder = Model(input=[decoder.input], output=[decoder_top.output])
        autoencoder = Model(input=[encoder.input], output=[decoder.output])
    ("Printing results for level: {}".format(level))
    print(autoencoder.summary())
    return 1

a = define_model(target_size=target_size, level=1)
a = define_model(target_size=target_size, level=2)
a = define_model(target_size=target_size, level=3)
a = define_model(target_size=target_size, level=4)
a = define_model(target_size=target_size, level=5)

# TODO: Set decoder accordingly to level:
# eg: level 5: load decoder for 4-3-2-1 and stack as so: encoder, level5, frozen-decoder-4-3-2-1
# eg: level 4: load encoder up to level 4, set-up trainable level 4 decoder, pass to frozen-decoder-3-2-1


autoencoder = Model(input=[encoder.input], output=[decoder])
autoencoder.summary()
