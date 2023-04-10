from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics, Dataset


project_id = 'qwiklabs-gcp-03-6e0d35a97dd4'
#pipeline_root_path = 'gs://tfds-dir1'
pipeline_root_path = 'gs://pipeline-tester1'


@component(
packages_to_install = ["tensorflow", "tensorflow-datasets"],
output_component_file="ingest_data.yaml"
)
def ingest_data() -> str:
    import tensorflow_datasets as tfds
    #bucket = 'gs://tfds-dir1'
    bucket = 'gs://pipeline-tester1'

    validation_split = 10
    #test_ds, cifar10_info = tfds.load('cifar10', split='test', with_info=True, as_supervised=True, shuffle_files=True, data_dir="gs://tfds-dir")
    test_ds = tfds.load('cifar10', split='test', as_supervised=True, shuffle_files=True, data_dir=bucket + "/test", try_gcs=True)

    validation_ds = tfds.load('cifar10', split=f'train[:{validation_split}%]', as_supervised=True, data_dir=bucket + "/valid", try_gcs=True)
    training_ds = tfds.load('cifar10', split=f'train[{validation_split}%:]', as_supervised=True, data_dir=bucket + "/train", try_gcs=True)

    #print(f'Classes:{cifar10_info.features["label"].names}')
    print(test_ds.element_spec)

    return bucket

@component(
packages_to_install = ["tensorflow"],
output_component_file="create_model.yaml"
)
def create_model(text: str) -> str:
    import tensorflow as tf
    #bucket = 'gs://tfds-dir/test_ds1'
    bucket = 'gs://pipeline-tester1'
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    new_model.save(bucket+"/model")
    return "model saved at" + bucket


@component(
packages_to_install = ["tensorflow",  "tensorflow-datasets"],
output_component_file="train_model.yaml"
)
def train_model(test: str) -> str:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    #bucket = 'gs://tfds-dir/test_ds1'
    bucket = 'gs://pipeline-tester1'

    #load data from gcs bucket
    model_name = 'ResNet model'
    train_data = tfds.load(bucket+"/train/cifar10/")
    train_data = train_data.map(lambda f, l: (tf.cast(f,tf.float64) / 255, l))
    train_data = train_data.shuffle(buffer_size=5000)

    valid_data = tfds.load(bucket+"/valid/cifar10/")
    #test_data = tfds.load("gs://tfds-dir/test_ds")

    #load model from gcs
    model = tf.keras.models.load_model(bucket+'/model')

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath= bucket+'/cifar10-{model_name}-'+ '{epoch:02d}-{val_accuracy:.4f}')

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])

    return "model trained"



@pipeline(
    name='pipeline',
    description='testing pipeline',
    pipeline_root=pipeline_root_path
)
def model_test():
    ingestion_task = ingest_data()
    create_model_task = create_model(ingestion_task.output)
    train_model_task = train_model(create_model_task.output)


if __name__ == '__main__':

    compiler.Compiler().compile(
        pipeline_func=model_test,
        package_path='pipeline.json'
    )
