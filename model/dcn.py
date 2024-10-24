import random
import sys
import csv
import gzip
import copy
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from tabulate import tabulate

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

import matplotlib.pyplot as plt


class DCN(tfrs.Model):

  def __init__(self, 
      dcn_parallel,
      cross_layer_size, 
      deep_layer_sizes, 
      vocabularies,
      str_features,
      int_features,
      embedding_dimension=32, 
      projection_dim=None):
    super().__init__()

    self.dcn_parallel = dcn_parallel
    self.embedding_dimension = embedding_dimension

    self._embedding_features = str_features
    self._int_features = int_features
        
    self._embeddings = {}
    self._vocabularies = vocabularies

    if 'one-hot encoding' in vocabularies:
      for feature_name, voca in vocabularies['one-hot encoding'].items():
        self._embeddings[feature_name] = tf.keras.layers.StringLookup(
                vocabulary=voca, mask_token=None, output_mode='one_hot')

    if 'top-n + one-hot encoding' in vocabularies:
      for feature_name, voca in vocabularies['top-n + one-hot encoding'].items():
        self._embeddings[feature_name] = tf.keras.layers.StringLookup(
                vocabulary=voca, mask_token=None, output_mode='one_hot')

    if 'threshold + embedding' in vocabularies:
      for feature_name, voca in vocabularies['threshold + embedding'].items():
        self._embeddings[feature_name] = tf.keras.Sequential(
            [tf.keras.layers.StringLookup(
                vocabulary=voca, mask_token=None),
             tf.keras.layers.Embedding(len(voca) + 1,
                                       self.embedding_dimension)
      ])

    if 'embedding' in vocabularies:
      for feature_name, voca in vocabularies['embedding'].items():
        self._embeddings[feature_name] = tf.keras.Sequential(
            [tf.keras.layers.StringLookup(
                vocabulary=voca, mask_token=None),
             tf.keras.layers.Embedding(len(voca) + 1,
                                       self.embedding_dimension)
      ])

    if cross_layer_size == 0:
      self._cross_layers = None
    else:
      self._cross_layers = [tfrs.layers.dcn.Cross(
              projection_dim=projection_dim,
              kernel_initializer="glorot_uniform")
          for layer_size in range(cross_layer_size)]

    if deep_layer_sizes == []:      
      self._deep_layers = None
    else:
      self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
        for layer_size in deep_layer_sizes]

    self._logit_layer = tf.keras.Sequential(
          [tf.keras.layers.Dense(1),
            tf.keras.layers.Activation(activation='sigmoid')
        ]
      )

    self.task = tfrs.tasks.Ranking(
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.BinaryCrossentropy(name="LogLoss"),
               tf.keras.metrics.AUC(name="AUC")]
    )

  def call(self, features):
    # Concatenate embeddings
    embeddings = []

    if 'one-hot encoding' in self._vocabularies:
      for feature_name in self._vocabularies['one-hot encoding'].keys():
        embed_v = self._embeddings[feature_name](features[feature_name])
        embeddings.append(tf.cast(embed_v, tf.float32))

    if 'top-n + one-hot encoding' in self._vocabularies:
      for feature_name in self._vocabularies['top-n + one-hot encoding'].keys():
        embed_v = self._embeddings[feature_name](features[feature_name])
        embeddings.append(tf.cast(embed_v, tf.float32))      

    if 'threshold + embedding' in self._vocabularies:
      for feature_name in self._vocabularies['threshold + embedding'].keys():
        embed_v = self._embeddings[feature_name](features[feature_name])
        embeddings.append(embed_v)      

    if 'embedding' in self._vocabularies:
      for feature_name in self._vocabularies['embedding'].keys():
        embed_v = self._embeddings[feature_name](features[feature_name])
        embeddings.append(embed_v)      

    for feature_name in self._int_features:
        expanded = tf.expand_dims(features[feature_name], axis=-1)
        embeddings.append(expanded)
        
    x0 = tf.concat(embeddings, axis=1)

    if self.dcn_parallel:
      if self._cross_layers is not None:
        cross_x = self._cross_layers[0](x0, x0)
        for cross_layer in self._cross_layers[1:]:
            cross_x = cross_layer(x0, cross_x)

      if self._deep_layers is not None:
        deep_x = self._deep_layers[0](x0)
        for deep_layer in self._deep_layers[1:]:
          deep_x = deep_layer(deep_x)

      if self._cross_layers is None:
        x = deep_x
      elif self._deep_layers is None:
        x = cross_x
      else:
        x = tf.concat([cross_x, deep_x], axis=1)

    else:
      if self._cross_layers is not None:
          x = self._cross_layers[0](x0, x0)
          for cross_layer in self._cross_layers[1:]:
              x = cross_layer(x0, x)
      else:
        x = x0

      if self._deep_layers is not None:
        for deep_layer in self._deep_layers:
          x = deep_layer(x)

    return tf.reshape(self._logit_layer(x), [-1])

  def compute_loss(self, features, training=False):
    labels = features['click']

    if labels.dtype == tf.string:
      labels = tf.strings.to_number(labels, out_type=tf.float32)
    else:
      labels = tf.cast(labels, tf.float32)

    scores = self(features)

    return self.task(
        labels=labels,
        predictions=scores,
    )