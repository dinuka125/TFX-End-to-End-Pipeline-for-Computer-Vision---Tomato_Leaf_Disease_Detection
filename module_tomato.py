import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os 

#tfx library imports
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.tfxio import dataset_options



_IMAGE_KEY = 'raw_image'
_LABEL_KEY = 'label'

def preprocessing_fn(inputs):
    """
    transform image data 
    """
    outputs = {}

    raw_image_dataset = inputs[_IMAGE_KEY]
    raw_image_dataset = tf.map_fn(fn = lambda x : tf.io.parse_tensor(x[0], tf.uint8, name=None), elems = raw_image_dataset, fn_output_signature = tf.TensorSpec((250,250,1),dtype=tf.uint8,    name=None), infer_shape = True)
    raw_image_dataset = tf.cast(raw_image_dataset, tf.float32)
    image_features = raw_image_dataset/255.0
         
    outputs[_IMAGE_KEY] = image_features
    outputs[_LABEL_KEY] = tf.cast(inputs[_LABEL_KEY], tf.int64)

    return outputs



def _build_keras_model() -> tf.keras.Model:
    """
    Creates a Image Classification model with Keras

    Retruns:
         The Image classification Keras Model 
    """    
    data_augmentation = keras.Sequential([

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])

    inputs = keras.Input(shape=(250,250,1), name='raw_image')
    x = data_augmentation(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.summary()
    return model




def _get_serve_image_fn(model, tf_transform_output):
    """ Return a function that feeds the input tensor into the model"""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_image_fn(image_tensor):
        return(model(image_tensor))
      

    return serve_image_fn    




def _input_fn(file_pattern, 
              data_accessor: DataAccessor, 
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """
    Generates features and labels for training.

    Args:
        file_pattern: List of paths or pattersn of input tfrecord files.
        data_accessor: DataAccessor for converting input to RecordBatch.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features is a 
        dictionary of Tensors, and indices is a single Tensor of lable indices.
    """

    dataset = data_accessor.tf_dataset_factory(
        file_pattern, 
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_LABEL_KEY),
            tf_transform_output.transformed_metadata.schema
        )
    return dataset     
   


def run_fn(fn_args: FnArgs):
    """Train the model base on given args.
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    Raises:
        ValueError: if invalid inputs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size= 32
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size= 32
    )

    model = _build_keras_model()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )

    signatures = {
        'serving_default':
        _get_serve_image_fn(model,tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None,250,250,1],
                dtype=tf.float32,
                name=(_IMAGE_KEY)))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)        


