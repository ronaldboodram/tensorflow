from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset


project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'
#pipeline_root_path = 'gs://tfds-dir/test_ds'
pipeline_root_path = 'gs://pipeline-tester'

@component(
packages_to_install = ["tensorflow", "tensorflow-datasets"]
)
def ingest_data() -> str:
    import tensorflow_datasets as tfds

    validation_split = 10
    bucket = 'gs://pipeline-tester'
    #test_ds, cifar10_info = tfds.load('cifar10', split='test', with_info=True, as_supervised=True, shuffle_files=True, data_dir="gs://tfds-dir")
    test_ds = tfds.load('cifar10', split='test', with_info=True, as_supervised=True, shuffle_files=True, data_dir=bucket)

    validation_ds = tfds.load('cifar10', split=f'train[:{validation_split}%]', as_supervised=True, data_dir=bucket)
    training_ds = tfds.load('cifar10', split=f'train[{validation_split}%:]', as_supervised=True, data_dir=bucket)

    #print(f'Classes:{cifar10_info.features["label"].names}')
    #print(test_ds.element_spec)

    return "gs://tfds-dir"

@component(
packages_to_install = ["tensorflow"]
)
def create_model() -> str:
    import tensorflow as tf

    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    new_model.save("gs://pipeline-tester/model")
    return "model saved at gcs://tfds-dir"





@pipeline(
    name='pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def ingestion_test():
    ingestion_task = ingest_data()
    create_model_task = create_model()


if __name__ == '__main__':

    compiler.Compiler().compile(
        pipeline_func=ingestion_test,
        package_path='pipeline.json'
    )
