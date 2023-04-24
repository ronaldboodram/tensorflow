from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset
#from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from typing import NamedTuple

# ***************************
# USES KUBEFLOW 2.0.0B14 SDK API TO CREATE CUSTOM TRAINING JOBS USING GPUs or CPUsWHERE EACH TASK IN A PIPELINE CAN HAVE
# ITS OWN HARDWARE CONFIGURATION
# KUBEFLOW SDK API DOES NOT SUPPORT TPU AS YET FOR V2.0.0B14. YOU HAVE TO USE KFP V1.8.20 OR EARLIER ALONG
# WITH KFP.GCP EXTENSION MODULE
# https://googlecloudplatform.github.io/kubeflow-gke-docs/docs/pipelines/enable-gpu-and-tpu/
# https://kubeflow-pipelines.readthedocs.io/en/1.8.20/source/kfp.extensions.html
# ***************************

project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'

#pipeline_root_path = 'gs://tfds-dir3'
pipeline_root_path = 'gs://pipeline-tester3'


# edit the pipeline.json file to remove the automatic install of kfp 1.8.9 which casues conflict with tensorflow 2.11
@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets', 'numpy==1.21.6']
)
def ingest_data() -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    from keras import datasets
    import tensorflow as tf

    validation_split = 10
    #bucket = 'gs://tfds-dir3'
    bucket = 'gs://pipeline-tester3/keras_model'

    #tfds.load('cifar10'

    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


   # need the "self" parameter as their is an implicit argument in the custom_shard_func
   #  that gives an error saying one arg expected but two were given
    def custom_shard_func(self, element):
        return np.int64(0)

    ds_train.save(path=bucket + "/ds_train", shard_func=custom_shard_func)
    ds_test.save(path=bucket + "/ds_test", shard_func=custom_shard_func)

    return bucket


@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'keras'],
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.2-11',
)
def create_model(text: str) -> str:
    import tensorflow as tf
    from keras import models, layers, datasets, applications

    #bucket = 'gs://tfds-dir3'
    bucket = 'gs://pipeline-tester3/keras_model'

    # check for GPU:
    print('\n\n GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print('\n\n')

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('\n\n' + str(model.summary()) + '\n\n')
    model.save(bucket + "/untrained-model")
    return "model saved:" + bucket


@dsl.component(
    packages_to_install=['tensorflow==2.11.0', 'tensorflow-datasets'],
    output_component_file="train_model.yaml",
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.2-11',
)
def train_model(text: str) -> str:
    import tensorflow_datasets as tfds
    import tensorflow as tf


    # check for GPU:
    print('\n\n GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print('\n\n')

    # env variable that tells tensorboard where to store logs
    #os.environ['AIP_TENSORBOARD_LOG_DIR']
    #os.environ['gs://tensorboard3']

    #Storage buckets
    # bucket = 'gs://tfds-dir3'
    bucket = 'gs://pipeline-tester3/keras_model'

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    # batch size
    # batch_size = 32 * strategy.num_replicas_in_sync

    # load data from gcs bucket
    ds_train = tf.data.Dataset.load(bucket + "/ds_train")
    ds_test = tf.data.Dataset.load(bucket + "/ds_test")

    print("\n\n finish loading training data")
    print("\n\n finish loading test data \n\n")

    # load model from gcs
    model = tf.keras.models.load_model(bucket + '/untrained-model')

    # Create and Train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    history = model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_test,
    )

    print("\n\n history" + str(history))

    #Save the model
    model.save(bucket + "/keras_ model")

    return "model trained"


@dsl.pipeline(
    name='keras_model_pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def keras_model_pipeline():
    ingestion_task = ingest_data()
    create_model_task = create_model(text=ingestion_task.output).set_accelerator_type('NVIDIA_TESLA_V100').set_cpu_limit('4').set_memory_limit('16G').set_accelerator_limit(2)
    train_model_task_4_V100_GPUs = train_model(text=create_model_task.output)\
        .set_accelerator_type('NVIDIA_TESLA_V100')\
        .set_cpu_limit('4')\
        .set_memory_limit('16G')\
        .set_accelerator_limit(2)\
        .set_display_name('2 x V100 GPUS ')
    # train_model_task_2_V100_GPUs = train_model(text=create_model_task.output)\
    #     .set_accelerator_type('NVIDIA_TESLA_V100')\
    #     .set_cpu_limit('4')\
    #     .set_memory_limit('16G')\
    #     .set_accelerator_limit(1)\
    #     .set_display_name('1 x V100 GPUS ')


if __name__ == '__main__':

        compiler.Compiler().compile(
        pipeline_func=keras_model_pipeline,
        package_path='keras-model-pipeline.json'
    )

