from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset

project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'
# pipeline_root_path = 'gs://tfds-dir1'
pipeline_root_path = 'gs://pipeline-tester1'

@component(
    packages_to_install=["tensorflow", "tensorflow-datasets"],
    output_component_file="train_model.yaml"
)
def train_model() -> str:
    import tensorflow_datasets as tfds
    import numpy as np
    import tensorflow as tf

    # bucket = 'gs://tfds-dir/test_ds1'
    bucket = 'gs://pipeline-tester1'

    # load data from gcs bucket
    model_name = 'ResNet model'
    train_data = tf.data.Dataset.load(bucket + "/train_ds")
    train_data = train_data.map(lambda f, l: (tf.cast(f, tf.float64) / 255, l))
    train_data = train_data.shuffle(buffer_size=5000)
    print("\n\n finish loading training data \n\n")

    valid_data = tf.data.Dataset.load(bucket + "/valid_ds")
    print("\n\n finish loading validation data \n\n")
    test_data = tf.data.Dataset.load(bucket + "/test_ds")
    print("\n\n finish loading test data \n\n")

    # load model from gcs
    model = tf.keras.models.load_model(bucket + '/model')
    print("\n\n finish loading model from gcs \n\n")

    # Create training callbacks
    # earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=bucket + '/cifar10-{model_name}-' + '{epoch:02d}-{val_accuracy:.4f}')

    # Train the model
    # history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])
    history = model.fit(train_data, validation_data=valid_data, epochs=10)

    return "model trained"
@pipeline(
    name='pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def ingestion_test():
        train_model_task = train_model()


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=ingestion_test,
        package_path='pipeline.json'
    )
