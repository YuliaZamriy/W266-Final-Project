#!/usr/bin/env python

# Source: https://developers.google.com/machine-learning/guides/text-classification/step-4

import time
import os
import numpy as np

import tensorflow as tf

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import GlobalMaxPooling1D


# from tensorflow.python.keras import initializers
# from tensorflow.python.keras import regularizers


# Custom Callbacks

class TimeHistory(tf.keras.callbacks.Callback):
  """
  https://stackoverflow.com/questions/43178668/
  record-the-computation-time-for-each-epoch-in-keras-during-model-fit
  """

  def on_train_begin(self, logs={}):
    self.times = []

  def on_epoch_begin(self, epoch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, epoch, logs={}):
    self.times.append(time.time() - self.epoch_time_start)

# Pre-trained embeddings


def build_emb_matrix(glove_dir=None,
                     word_index=None,
                     emb_dim=100,
                     max_num_words=20000):

  emb_filename = 'glove.6B.' + str(emb_dim) + 'd.txt'

  start = time.time()

  embeddings_index = {}
  with open(os.path.join(glove_dir, emb_filename)) as f:
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

  stage1 = time.time()
  print('Found %s word vectors.' % len(embeddings_index))
  print('It took {0:.1f} seconds'.format(stage1 - start))

  print('Preparing embedding matrix.')

  num_words = min(max_num_words, len(word_index))
  embedding_matrix = np.zeros((num_words, emb_dim))
  for word, i in word_index.items():
    if i > max_num_words - 1:
      break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all-zeros.
      embedding_matrix[i] = embedding_vector

  stage2 = time.time()
  print('Embedding matrix has been built.')
  print('Its shape is {}.'.format(embedding_matrix.shape))
  print('It took {0:.1f} seconds'.format(stage2 - stage1))

  return embedding_matrix

# NN Ngram Model


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
  """
  Creates an instance of a multi-layer perceptron model.

  Arguments
      layers: int, number of `Dense` layers in the model.
      units: int, output dimension of the layers.
      dropout_rate: float, percentage of input to drop at Dropout layers.
      input_shape: tuple, shape of input to the model.
      num_classes: int, number of output classes.

  Returns
      An MLP model instance.
  """
  op_units, op_activation = 1, 'sigmoid'
  model = models.Sequential()
  model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

  for _ in range(layers - 1):
    model.add(Dense(units=units, activation='relu'))
    model.add(Dropout(rate=dropout_rate))

  model.add(Dense(units=op_units, activation=op_activation))

  return model


# CNN Model


def cnn_model(filters,
              kernel_size,
              layers,
              embedding_dim,
              dropout_rate,
              pool_size,
              input_shape,
              num_features,
              use_pretrained_embedding=False,
              is_embedding_trainable=False,
              glove_dir=None,
              word_index=None):

  op_units, op_activation = 1, 'sigmoid'
  model = models.Sequential()

  if use_pretrained_embedding:
    embedding_matrix = build_emb_matrix(glove_dir,
                                        word_index,
                                        embedding_dim,
                                        num_features)
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_shape[0],
                        trainable=is_embedding_trainable))
  else:
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

  for i in range(layers):
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu',
                     bias_initializer='random_uniform',
                     padding='same'))
    if i == layers - 1:
      model.add(GlobalMaxPooling1D())
    else:
      model.add(MaxPooling1D(pool_size=pool_size))

  model.add(Dropout(rate=dropout_rate))
  model.add(Dense(op_units, activation=op_activation))

  return model

# sepCNN Model


def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 glove_dir=None,
                 word_index=None):
  """
  Source: https://github.com/google/eng-edu/blob/master/ml/guides/text_classification/build_model.py

  Creates an instance of a separable CNN model.

  # Arguments
      blocks: int, number of pairs of sepCNN and pooling blocks in the model.
      filters: int, output dimension of the layers.
      kernel_size: int, length of the convolution window.
      embedding_dim: int, dimension of the embedding vectors.
      dropout_rate: float, percentage of input to drop at Dropout layers.
      pool_size: int, factor by which to downscale input at MaxPooling layer.
      input_shape: tuple, shape of input to the model.
      num_classes: int, number of output classes.
      num_features: int, number of words (embedding input dimension).
      use_pretrained_embedding: bool, true if pre-trained embedding is on.
      is_embedding_trainable: bool, true if embedding layer is trainable.
      embedding_matrix: dict, dictionary with embedding coefficients.

  # Returns
      A sepCNN model instance.
  """

  # adjust this if using more than two classes in target
  op_units, op_activation = 1, 'sigmoid'
  model = models.Sequential()

  # Add embedding layer. If pre-trained embedding is used add weights to the
  # embeddings layer and set trainable to input is_embedding_trainable flag.
  if use_pretrained_embedding:
    embedding_matrix = build_emb_matrix(glove_dir,
                                        word_index,
                                        embedding_dim,
                                        num_features)
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0],
                        weights=[embedding_matrix],
                        trainable=is_embedding_trainable))
  else:
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

  for _ in range(blocks - 1):
    model.add(Dropout(rate=dropout_rate))
    model.add(SeparableConv1D(filters=filters,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(MaxPooling1D(pool_size=pool_size))

  model.add(SeparableConv1D(filters=filters * 2,
                            kernel_size=kernel_size,
                            activation='relu',
                            bias_initializer='random_uniform',
                            depthwise_initializer='random_uniform',
                            padding='same'))
  model.add(SeparableConv1D(filters=filters * 2,
                            kernel_size=kernel_size,
                            activation='relu',
                            bias_initializer='random_uniform',
                            depthwise_initializer='random_uniform',
                            padding='same'))
  model.add(GlobalAveragePooling1D())
  model.add(Dropout(rate=dropout_rate))
  model.add(Dense(op_units, activation=op_activation))
  return model

# Train model


def train_model(data,
                word_index=None,
                log_dir=None,
                model='cnn',
                learning_rate=1e-3,
                epochs=1000,
                batch_size=128,
                blocks=2,
                filters=64,
                layers=2,
                units=64,
                dropout_rate=0.2,
                embedding_dim=200,
                kernel_size=3,
                pool_size=3,
                max_num_words=20000,
                use_pretrained_embedding=False,
                is_embedding_trainable=False,
                glove_dir=None):
  """Trains sequence model on the given dataset.
  # Arguments
      data: tuples of vectorized training and test texts and labels.
      learning_rate: float, learning rate for training model.
      epochs: int, number of epochs.
      batch_size: int, number of samples per batch.
      blocks: int, number of pairs of sepCNN and pooling blocks in the model.
      filters: int, output dimension of sepCNN layers in the model.
      dropout_rate: float: percentage of input to drop at Dropout layers.
      embedding_dim: int, dimension of the embedding vectors.
      kernel_size: int, length of the convolution window.
      pool_size: int, factor by which to downscale input at MaxPooling layer.
  # Raises
      ValueError: If validation data has label values which were not seen
          in the training data.
  """
  # Get the data.
  (x_train, train_labels), (x_val, val_labels) = data

  # Verify that validation labels are in the same range as training labels.
  num_classes = 2

  # Number of features will be the embedding input dimension. Add 1 for the
  # reserved index 0.

  if word_index:
    num_features = min(len(word_index) + 1, max_num_words)

  # Create model instance.
  if model == 'cnn':
    model = cnn_model(filters=filters,
                      kernel_size=kernel_size,
                      layers=layers,
                      embedding_dim=embedding_dim,
                      dropout_rate=dropout_rate,
                      pool_size=pool_size,
                      input_shape=x_train.shape[1:],
                      num_features=num_features,
                      use_pretrained_embedding=use_pretrained_embedding,
                      glove_dir=glove_dir,
                      word_index=word_index)
  elif model == 'sepcnn':
    model = sepcnn_model(blocks=blocks,
                         filters=filters,
                         kernel_size=kernel_size,
                         embedding_dim=embedding_dim,
                         dropout_rate=dropout_rate,
                         pool_size=pool_size,
                         input_shape=x_train.shape[1:],
                         num_features=num_features,
                         use_pretrained_embedding=use_pretrained_embedding,
                         glove_dir=glove_dir,
                         word_index=word_index)
  elif model == 'ngram':
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)

  # Compile model with learning parameters.
  optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
  model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['acc'])

  # Create callback for early stopping on validation loss. If the loss does
  # not decrease in two consecutive tries, stop training.
  callbacks = [
      tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=2),
      tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True),
      TimeHistory()]

  # Train and validate model.
  history = model.fit(
      x_train,
      train_labels,
      epochs=epochs,
      callbacks=callbacks,
      validation_data=(x_val, val_labels),
      verbose=2,  # Logs once per epoch.
      batch_size=batch_size)

  # Print results.
  history = history.history
  print('Validation accuracy: {acc}, loss: {loss}'.format(
      acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

  train_pred_probs = model.predict(x_train)
  val_pred_probs = model.predict(x_val)

  return history, train_pred_probs, val_pred_probs
