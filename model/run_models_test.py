import random
import sys
import csv
import gzip
import copy
import datetime
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from tabulate import tabulate

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

import matplotlib.pyplot as plt
import dcn
def run_models(
  dcn_parallel,
  cross_layer_size, 
  deep_layer_sizes, 
  vocabularies,
  str_features,
  int_features,
  train,
  test,
  learning_rate,
  epochs,
  history_file_dir,
  history_file_name,
  checkpoints_dir,
  checkpoints_name,
  embedding_dimension=32,
  projection_dim=None, 
  num_runs=5):

  fitting_history = {}

  for num_run in range(num_runs):
    fitting_data = {
      'Train LogLoss' : [],
      'Train AUC' : [],
      'Test LogLoss' : [],
      'Test AUC' : []
    }
    model = dcn.DCN(
            dcn_parallel,
            cross_layer_size,
            deep_layer_sizes,
            vocabularies,
            str_features,
            int_features,                
            embedding_dimension)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    for epoch in range(epochs):
      history = model.fit(train, epochs=1, verbose=True)
      metrics = model.evaluate(test, return_dict=True)
      return history, metrics
      fitting_data['Train LogLoss'].append(history["LogLoss"][0])
      fitting_data['Train AUC'].append(history["AUC"][0])
      fitting_data['Test LogLoss'].append(metrics["LogLoss"])
      fitting_data['Test AUC'].append(metrics["AUC"])
      if (epoch % 10) == 0:
          print("{}th Epoch".format(epoch+1))
          with open(history_file_dir + history_file_name + '_num_' + str(num_run) + '.p', 'wb') as f:
              pickle.dump(fitting_data, f)
          model.save_weights(checkpoints_dir + checkpoints_name + '_num_' + str(num_run) + '.weights.h5')
    fitting_history[num_run] = fitting_data
  return fitting_history