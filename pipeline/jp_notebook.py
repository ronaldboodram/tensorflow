import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json
import time

# from history import plot_history, add_history


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save_model(model, name, history, test_data):
    # bucket = 'gs://tfds-dir/test_ds1'
    bucket = 'gs://pipeline-tester1'

    test_loss, test_acc = model.evaluate(test_data)

    # To set the decimal precision after the comma,
    # you can use the f-string formatting functionality in Python 3.
    # For example, the expression f'{x:.3f}' converts the float variable x to a float with precision 3.

    # Save model information
    save_name = bucket + f'/models/cifar10-{name}-{len(history.epoch):02d}-{test_acc * 100:.2f}'
    model.save(f'{save_name}.h5')

    # Save history information
    hist_out = {}
    hist_out['epoch'] = history.epoch
    hist_out['history'] = history.history
    hist_out['params'] = history.params
    with open(f'{save_name}.history', 'w') as outfile:
        json.dump(hist_out, outfile)


if __name__ == '__main__':
    # bucket = 'gs://tfds-dir/test_ds1'
    bucket = 'gs://pipeline-tester1'

    batch_size = 32

    # load data from gcs bucket
    model_name = 'ResNet'
    train_data = tf.data.Dataset.load(bucket + "/train_ds")
    train_data = train_data.map(lambda f, l: (tf.cast(f, tf.float64) / 255, l))
    train_data = train_data.shuffle(buffer_size=5000)
    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("\n\n finish loading training data \n\n")

    valid_data = tf.data.Dataset.load(bucket + "/valid_ds")
    valid_data = valid_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("\n\n finish loading validation data \n\n")
    test_data = tf.data.Dataset.load(bucket + "/test_ds")
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("\n\n finish loading test data \n\n")

    # load model from gcs
    model = tf.keras.models.load_model(bucket + '/model')
    print("\n\n finish loading model from gcs \n\n")

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=bucket + f'/ckpts/cifar10-{model_name}-' + '{epoch:02d}-{val_accuracy:.4f}')

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

    # Save model information
    save_model(model, model_name, history, test_data)
