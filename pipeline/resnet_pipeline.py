from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'
pipeline_root_path = 'gs://tfds-dir2'
# pipeline_root_path = 'gs://pipeline-tester2'


# edit the pipeline.json file to remove the automatic install of kfp 1.8.9 which casues conflict with tensorflow 2.11
@component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets', 'numpy==1.21.6']
)
def ingest_data() -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    print("\n\n" + "Tensorflow version is: " + tf.__version__ + "\n\n")
    print("\n\n" + "Tfds version is: " + tfds.__version__ + "\n\n")

    validation_split = 10
    bucket = 'gs://tfds-dir2'
    # bucket = 'gs://pipeline-tester2'

    # test_ds, cifar10_info = tfds.load('cifar10', split='test', with_info=True, as_supervised=True, shuffle_files=True, data_dir="gs://tfds-dir")
    # test_ds = tfds.load('cifar10', split='test', as_supervised=True, shuffle_files=True, data_dir=bucket + "/test", try_gcs=True)
    # validation_ds = tfds.load('cifar10', split=f'train[:{validation_split}%]', as_supervised=True, data_dir=bucket + "/valid", try_gcs=True)
    # training_ds = tfds.load('cifar10', split=f'train[{validation_split}%:]', as_supervised=True, data_dir=bucket + "/train", try_gcs=True)

    test_ds = tfds.load('cifar10', split='test', as_supervised=True, shuffle_files=True)
    validation_ds = tfds.load('cifar10', split=f'train[:{validation_split}%]', as_supervised=True)
    training_ds = tfds.load('cifar10', split=f'train[{validation_split}%:]', as_supervised=True)

    # need the "self" parameter as their is an implicit argument in the custom_shard_func
    # that gives an error saying one arg expected but two were given
    def custom_shard_func(self, element):
        return np.int64(0)

    training_ds.save(
        path=bucket + "/train_ds", shard_func=custom_shard_func)

    validation_ds.save(
        path=bucket + "/valid_ds", shard_func=custom_shard_func)

    test_ds.save(
        path=bucket + "/test_ds", shard_func=custom_shard_func)

    return bucket


@component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets', 'numpy==1.21.6']
)
def load_data(text: str) -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    bucket = 'gs://tfds-dir2'
    # bucket = 'gs://pipeline-tester2'

    new_dataset = tf.data.Dataset.load(bucket + "/train1")

    return "loaded"


@component(
    packages_to_install=['tensorflow==2.11.0', 'keras']
)
def create_model(test: str) -> str:
    import tensorflow as tf
    from keras import applications

    bucket = 'gs://tfds-dir2'
    # bucket = 'gs://pipeline-tester2'

    new_model = tf.keras.Sequential([
        applications.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    new_model.save(bucket + "/model")
    return "model saved:" + bucket


@component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets'],
    output_component_file="train_model.yaml"
)
def train_model(text: str) -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    bucket = 'gs://tfds-dir2'
    # bucket = 'gs://pipeline-tester2'

    # batch size
    batch_size = 32

    # load data from gcs bucket
    model_name = 'ResNet model'
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

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=bucket + f'/ckpts/cifar10-{model_name}-' + '{epoch:02d}-{val_accuracy:.4f}')

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])
    #history = model.fit(train_data, validation_data=valid_data, epochs=10)
    print('\n\n history\n' + history + '\n\n')

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print('\n\n'  + f'Test accuracy: {test_acc * 100:.2f}%' + '\n\n')

    #Save the model
    model.save(bucket + "/resnet_ model")

    return "model trained"
@pipeline(
    name='pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def ingestion_test():
    ingestion_task = ingest_data()
    # load_data_task = load_data(ingestion_task.output)
    create_model_task = create_model(ingestion_task.output)
    train_model_task = train_model(create_model_task.output)


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=ingestion_test,
        package_path='pipeline.json'
    )
