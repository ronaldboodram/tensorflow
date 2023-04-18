from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'
# pipeline_root_path = 'gs://tfds-dir2'
pipeline_root_path = 'gs://pipeline-tester2'

@dsl.component(
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.2-11',
    packages_to_install=["tensorflow==2.11.0", "tensorflow-datasets"],
    output_component_file="train_model.yaml"
)
def train_model() -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    #check for GPU:
    print('\n\n GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    print('\n\n')

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    #Storage buckets
    bucket = 'gs://tfds-dir3'
    # bucket = 'gs://pipeline-tester3'

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


    with strategy.scope():
        # Train the model
        # history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])
        # history = model.fit(train_data, validation_data=valid_data, epochs=10)
        # print('\n\n history\n' + str(history) + '\n\n')
        model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])


        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_data)
        print('\n\n' + f'Test accuracy: {test_acc * 100:.2f}%' + '\n\n')

    # Save the model
    model.save(bucket + "/resnet_ model")

    return "model trained"


#convert above component into a cusotm training job
custom_training_job = create_custom_training_job_from_component(
    train_model,
    display_name = 'Training Op',
    machine_type = 'n1-standard-16',
    accelerator_type='NVIDIA_TESLA_K80',
    accelerator_count='2'
)

@dsl.pipeline(
    name='pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def pipeline():
    train_model_task = train_model().set_accelerator_type('NVIDIA_TESLA_K80').set_cpu_limit('4').set_memory_limit('16G').set_accelerator_limit(2)
    # train_model_task = custom_training_job(
    #     project='tensor-1-1',
    #     location='us-central1'
    # )


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='training_op_pipeline.json'
    )
