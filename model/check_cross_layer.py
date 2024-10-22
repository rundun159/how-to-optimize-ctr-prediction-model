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

def load_model(
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
  projection_dim=None):

  model = dcn.DCN(
          dcn_parallel,
          cross_layer_size,
          deep_layer_sizes,
          vocabularies,
          str_features,
          int_features,                
          embedding_dimension)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

  num_run = 0
  model.load_weights(checkpoints_dir + checkpoints_name + '_num_' + str(num_run) + '.weights.h5')

  return model