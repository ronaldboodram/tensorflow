#%matplotlib inline
#import cv2
#from resnets_utils import *

import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from keras import applications
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

def retrieve_data():
    test_ds, cifar10_info = tfds.load('cifar10', split='test', with_info=True, as_supervised=True,shuffle_files=True)
    print(cifar10_info)
    print(f'Classes:{cifar10_info.features["label"].names}')
    print(test_ds.element_spec)

    return test_ds, cifar10_info

def get_training_data(validation_split=10):
    validation_ds = tfds.load('cifar10', split=f'train[:{validation_split}%]', as_supervised=True)
    training_ds = tfds.load('cifar10', split=f'train[{validation_split}%:]', as_supervised=True)

    return training_ds, validation_ds

def wrangle_data(data, split, batch_size=32):
    data = data.map(lambda f, l: (tf.cast(f,tf.float64) / 255, l))

    if split == 'train':
        data = data.shuffle(buffer_size=5000)
    elif split == 'valid':
        data = data.cache()
    elif split == 'test':
        data = data.cache()

    data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data

# Create a model that uses a ResNet model. add the output layer and the input layer to it.
def compile_model(new_model):
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    return new_model

def save_model(model, name, history, test_data):
    test_loss, test_acc = model.evaluate(test_data)

    # Save model information
    save_name = f'models/cifar10-{name}-{len(history.epoch):02d}-{test_acc * 100:.2f}'
    model.save(f'{save_name}.h5')

    # Save history information
    hist_out = {}
    hist_out['epoch'] = history.epoch
    hist_out['history'] = history.history
    hist_out['params'] = history.params
    with open(f'{save_name}.history', 'w') as outfile:
        json.dump(hist_out,outfile)

if __name__ == '__main__':
    # Prepare the data
    test_ds, image_info = retrieve_data()
    train_ds, valid_ds = get_training_data(validation_split=10)

    # Wrangle data
    batch_size = 64
    train_data = wrangle_data(train_ds, 'train', batch_size=batch_size)
    valid_data = wrangle_data(valid_ds, 'valid', batch_size=batch_size)
    test_data = wrangle_data(test_ds, 'test', batch_size=batch_size)

    #create ResNet model:
    base_model = applications.Reset50.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(10, activation='softmax')(x)
    model_name = 'resnet'
    model = Model(inputs=base_model.input, outputs=predictions)
    from keras.optimizers import Adam

    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(X_train, Y_train, epochs=100, batch_size=64)

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'ckpts/cifar10-{model_name}-'+ '{epoch:02d}-{val_accuracy:.4f}')


    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

    # Save model information
    save_model(model, model_name, history, test_data)
