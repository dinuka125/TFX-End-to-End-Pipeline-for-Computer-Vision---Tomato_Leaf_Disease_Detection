# TFX-End-to-End-Pipeline-for-Computer-Vision---Tomato_Leaf_Disease_Detection
This repository demonstrates the end-to-end workflow of Computer vision Classification based problem and the steps required to analyze, validate, and transform data, train a model, analyze its performance, and serve it. 

Used TFX Components 
   - ExampleGen ingests and splits the input dataset.
   - StatisticsGen calculates statistics for the dataset.
   - SchemaGen examines the statistics and creates a data schema.
   - ExampleValidator looks for anomalies and missing values in the dataset.
   - Transform performs feature engineering on the dataset.
   - Trainer trains the model using TensorFlow Estimators or Keras.
   - Evaluator performs deep analysis of the training results.
   - Pusher deploys the model to a serving infrastructure.
  
Tensorflow serving is used for the deployement   
  
