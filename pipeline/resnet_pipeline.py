from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset
#from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from typing import NamedTuple
project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'
pipeline_root_path = 'gs://tfds-dir3'
# pipeline_root_path = 'gs://pipeline-tester3'


# edit the pipeline.json file to remove the automatic install of kfp 1.8.9 which casues conflict with tensorflow 2.11
@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets', 'numpy==1.21.6']
)
def ingest_data() -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    print("\n\n" + "Tensorflow version is: " + tf.__version__ + "\n\n")
    print("\n\n" + "Tfds version is: " + tfds.__version__ + "\n\n")

    validation_split = 10
    bucket = 'gs://tfds-dir3'
    # bucket = 'gs://pipeline-tester3'

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

    bucket = 'gs://tfds-dir3'
    # bucket = 'gs://pipeline-tester3'

    new_dataset = tf.data.Dataset.load(bucket + "/train1")

    return "loaded"


@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'keras'],
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.2-11',
)
def create_model(text: str) -> str:
    import tensorflow as tf
    from keras import applications

    bucket = 'gs://tfds-dir3'
    # bucket = 'gs://pipeline-tester3'

    # check for GPU:
    print('\n\n GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print('\n\n')

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        new_model = tf.keras.Sequential([
            applications.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('\n\n' + str(new_model.summary()) + '\n\n')
    new_model.save(bucket + "/model")
    return "model saved:" + bucket


@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets'],
    output_component_file="train_model.yaml",
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.2-11',
)
def train_model(text: str) -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    # check for GPU:
    print('\n\n GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print('\n\n')

    #Storage buckets
    bucket = 'gs://tfds-dir3'
    # bucket = 'gs://pipeline-tester3'

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    # batch size
    batch_size = 32 * strategy.num_replicas_in_sync

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
    history = model.fit(train_data, validation_data=valid_data, epochs=30, callbacks=[earlystop, checkpoint])
    #history = model.fit(train_data, validation_data=valid_data, epochs=10)
    print('\n\n history\n' + str(history) + '\n\n')

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print('\n\n'  + f'Test accuracy: {test_acc * 100:.2f}%' + '\n\n')

    #Save the model
    model.save(bucket + "/resnet_ model")

    return "model trained"


#convert above component into a cusotm training job
# custom_create_model_job = create_custom_training_job_from_component(
#     train_model,
#     display_name = 'Create Model Op',
#     machine_type = 'n1-standard-16',
#     accelerator_type='NVIDIA_TESLA_K80',
#     accelerator_count='2'
# )

@dsl.pipeline(
    name='custom-container-pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def ingestion_test():
    ingestion_task = ingest_data()
    # load_data_task = load_data(ingestion_task.output)
    create_model_task = create_model(text=ingestion_task.output).set_accelerator_type('NVIDIA_TESLA_V100').set_cpu_limit('4').set_memory_limit('16G').set_accelerator_limit(4)
    #     text=ingestion_task.outputs['text'],
    #     project='tensor-1-1')
    # create_model_task = (
    #     create_model(text=ingestion_task.outputs['text']),
    #     set_cpu_limit('4'),
    #     set_memory_limit('16G'),
    #     add_node_selector_constraint('cloud.google.com/gke-accelerator', 'NVIDIA_TESLA_K80'),
    #     set_gpu_limit(2)
    # )
    train_model_task = train_model(text=create_model_task.output).set_accelerator_type('NVIDIA_TESLA_V100').set_cpu_limit('4').set_memory_limit('16G').set_accelerator_limit(4)


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=ingestion_test,
        package_path='pipeline.json'
    )
