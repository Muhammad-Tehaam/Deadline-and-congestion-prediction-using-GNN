"""
   Copyright 2020 Universitat Politecnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import tensorflow as tf
import configparser
import pandas as pd
import csv
import numpy as np
import tempfile
import os
from experimental_read_dataset import input_fn,return_corresponding_data
from Second_Experimental_RouteNet import model_fn
import random

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def train_and_evaluate(train_dir, eval_dir, config, model_dir=None):
    """Trains and evaluates the model.

    Args:
        train_dir (string): Path of the training directory.
        eval_dir (string): Path of the evaluation directory.
        config (configparser): Config file containing the diferent configurations
                               and hyperparameters.
        model_dir (string): Directory where all outputs (checkpoints, event files, etc.) are written.
                            If model_dir is not set, a temporary directory is used.
    """
    # csv_file = 'ground_truth_dataset.csv'
    # keys = ["Arrival Time","Duration","Volume","Source", "Destination", "Path", "Average Packets", "Average Bandwidth","Deadline","Remaining Time","Actual Delay", "Completion Time","jitter","Deadline Will be Met"]
    # with open(csv_file, 'w', newline='') as file:
    #     # Create a CSV writer object
    #         writer = csv.DictWriter(file, fieldnames=keys)
    #         writer.writeheader()


    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=int(config['RUN_CONFIG']['save_checkpoints_secs']),
        keep_checkpoint_max=int(config['RUN_CONFIG']['keep_checkpoint_max'])
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=my_checkpoint_config,
        params=config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_dir, repeat=True, shuffle=True),
        max_steps=int(config['RUN_CONFIG']['train_steps'])
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_dir, repeat=False, shuffle=False),
        throttle_secs=int(config['RUN_CONFIG']['throttle_secs'])
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def predict(test_dir, model_dir, config):
    """Generate the predictions given a model.

    Args:
        test_dir (string): Path of the test directory.
        model_dir (string): Directory with the trained model.
        config (configparser): Config file containing the diferent configurations
                               and hyperparameters.

    Returns:
        list: A list with the predicted values.
    """

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=config
    )
    i = 0
    csv_file = 'ground_truth_dataset.csv'
    keys = ["Arrival Time","Duration","Volume","Source", "Destination", "Path", "Average Packets", "Average Bandwidth","Deadline","Actual Delay", "Completion Time","Deadline Will be Met"]
    with open(csv_file, 'w', newline='') as file:
        # Create a CSV writer object
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
    pred_results = estimator.predict(input_fn=lambda: input_fn(test_dir, repeat=False, shuffle=False))#,generat_data = False)) # generat_data set to FALSE if the ground truth for multimodling is not required!
    results = list(pred_results)
    print(len(results)) 
    # test_data = return_corresponding_data(test_dir)
    # print(test_data[1])
    # print(len(test_data[1]))
    # with open("congestion_info.csv", 'w', newline='') as csv_file:
    #     # Create a CSV writer object
    #     csv_writer = csv.writer(csv_file)

    #     # Write the header row with dictionary keys
    #     header = ['links', 'Congestion Probability']
    #     csv_writer.writerow(header)

    #     # Write each key-value pair to the CSV file
    #     for link,cong in zip(test_data[1],results):
    #         csv_writer.writerow([link,cong['congestion']])

    # print(len(test_data))
 


    import numpy as np

      # Parameters
    lambda_value = 0.1  # Parameter for exponential distribution (controls mean arrival time)
    lower_limit_duration = 5  # Lower limit for duration
    upper_limit_duration = 30  # Upper limit for duration
    lower_limit_volume = 100  # Lower limit for volume
    upper_limit_volume = 1000  # Upper limit for volume
    available_nodes = [1, 2, 3, 4, 5]  
    
    num_flows = 54399  
    traffic_log = []

    for _ in range(num_flows):
        arrival_time = np.random.exponential(scale=1/lambda_value)
        duration = np.random.uniform(lower_limit_duration, upper_limit_duration)
        volume = np.random.uniform(lower_limit_volume, upper_limit_volume)
        deadline = np.random.uniform(20,40) 

        source_node = np.random.choice(available_nodes)
        destination_nodes = [node for node in available_nodes if node != source_node]
        destination_node = np.random.choice(destination_nodes)
          
        flow_info = [arrival_time, duration, volume, source_node, destination_node,deadline]
        traffic_log.append(flow_info)



# Open the CSV file in write mode
  #  with open(csv_file, 'w', newline='') as file:
        # Create a CSV writer object
   #     writer = csv.DictWriter(file, fieldnames=keys)

        # Write the header row
    #    writer.writeheader()

     #   for row, data,trafficLog in zip(pred_results, test_data,traffic_log):
           
      #      src, dst, path, top, avgP, avgbw = data
       #     AT,Dur,Vol,S,Dst,DL = trafficLog
        #    row_dict = {
         #       "delay": row["delay"],
          #      "jitter": row["jitter"],
           #     "Source": src,
            #    "Destination": dst,
             #   "Path": path,
               
              #  "Average Packets": avgP,
               # "Average Bandwidth": avgbw,
                #"Arrival Time": AT,
      #          "Duration": Dur,
       #         "Volume": Vol,
        #        "Deadline":DL
         #   }
          #  writer.writerow(row_dict)

            #simulate_network(src,dst,"0","0",path)


        # Iterate through the list of dictionaries and write the data
        
    print(f'Data written to {csv_file}')

      # for key,value in pred_obj.keys():
       #   print(key,":",value)
   








if __name__ == '__main__':
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read('../code/config.ini')


    train_and_evaluate(config['DIRECTORIES']['train'],
                       config['DIRECTORIES']['test'],
                      model_dir=config['DIRECTORIES']['logs'],config=config)
   # mre = predict_and_save(config['DIRECTORIES']['test'],
    #                       config['DIRECTORIES']['logs'],
     #                      '../dataframes/',
      #                    'predictions2.csv',
       #                   config._sections)
    mre = predict(config['DIRECTORIES']['test'],
                        config['DIRECTORIES']['logs'],
                           config._sections)
   # print(mre)