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

Module file includes the util functions  
  
#The dataset 
This example uses the kaggle tomato dataset (https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
   
 - Orchestrators - Apache Airflow and Apache beam    

The Airflow_pipe_tomato-Dag file.py includes the pipeline with - Airflow orchestrator configurations

The Airflow_pipe_tomato-Dag file.py includes the pipeline with - Airflow orchestrator configurations.

The TFX_Production_Pipeline_for_Tomato_Leaf_Disease_Detection includes the pipeline withe - Beam Orchestrator configurations
   
![Screenshot 2022-09-20 090514](https://user-images.githubusercontent.com/47025217/192094415-107d2a75-b2be-4b43-ad71-599dd80b6d5c.jpg)
