import pickle
import tensorflow as tf

# wget https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b432_vgg-100/vgg-100.zip
# unzip vgg-100.zip
# wget https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b634_resnet-100/resnet-100.zip
# unzip resnet-100.zip
# wget https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b498_inception-100/inception-100.zip
# unzip inception-100.zip

# apython3 feature_extraction.py --training_file vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg_cifar10_bottleneck_features_validation.p
# apython3 feature_extraction.py --training_file resnet_cifar10_100_bottleneck_features_train.p --validation_file resnet_cifar10_bottleneck_features_validation.p
# apython3 feature_extraction.py --training_file inception_cifar10_100_bottleneck_features_train.p --validation_file inception_cifar10_bottleneck_features_validation.p

# apython3 feature_extraction.py --training_file vgg_traffic_100_bottleneck_features_train.p --validation_file vgg_traffic_bottleneck_features_validation.p
# apython3 feature_extraction.py --training_file resnet_traffic_100_bottleneck_features_train.p --validation_file resnet_traffic_bottleneck_features_validation.p
# apython3 feature_extraction.py --training_file inception_traffic_100_bottleneck_features_train.p --validation_file inception_traffic_bottleneck_features_validation.p 

# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Activation
import numpy as np

EPOCHS = 50
BATCH_SIZE = 256

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    number_classes = len(np.unique(y_train))

    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(Dense(number_classes))
    model.add(Activation('softmax'))

    # TODO: train your model here
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=EPOCHS, shuffle=True, validation_data=(X_val, y_val))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
