from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset

project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'
# pipeline_root_path = 'gs://tfds-dir1/test_ds'
pipeline_root_path = 'gs://pipeline-tester1'


@component(
    packages_to_install=["tensorflow", "tensorflow-datasets"],
    output_component_file="train_model.yaml"
)
def train_model() -> str:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    # bucket = 'gs://tfds-dir/test_ds1'
    bucket = 'gs://pipeline-tester1'

    # load model from gcs
    model = tf.keras.models.load_model(bucket + '/model/')
    print("\n\n after loading model from gcs\n\n")

    # load data from gcs bucket
    model_name = 'ResNet model'
    train_data = tfds.load('train_data', data_dir= bucket + "/train")
    train_data = train_data.map(lambda f, l: (tf.cast(f, tf.float64) / 255, l))
    train_data = train_data.shuffle(buffer_size=5000)

    print("\n\n after train data\n\n")

    valid_data = tfds.load('https://storage.cloud.google.com/pipeline-tester1/train/cifar10/3.0.2/*')
    # test_data = tfds.load("gs://tfds-dir/test_ds")
    print("\n\n after valid data\n\n")

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=bucket + '/cifar10-{model_name}-' + '{epoch:02d}-{val_accuracy:.4f}')

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])

    return "model trained"

@pipeline(
    name='pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def model_test():
    train_model_task = train_model()


if __name__ == '__main__':

    compiler.Compiler().compile(
        pipeline_func=model_test,
        package_path='pipeline.json'
    )