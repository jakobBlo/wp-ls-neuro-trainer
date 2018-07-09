from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import json

class Mnist(object):
    def __init__(
        self,
        config,
        value_sets
        ):

        self.config = config
        self.value_sets = value_sets

        tf.logging.set_verbosity(tf.logging.INFO)

        # Load training and eval data
        self.mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = self.mnist.train.images  # Returns np.array
        self.train_labels = np.asarray(self.mnist.train.labels, dtype=np.int32)
        self.eval_data = self.mnist.test.images  # Returns np.array
        self.eval_labels = np.asarray(self.mnist.test.labels, dtype=np.int32)

    def set_config(self, config):
        self.config = config

    def cnn_model_fn(self, features, labels, mode):
      """Model function for CNN."""
      # Input Layer
      # Reshape X to 4-D tensor: [batch_size, width, height, channels]
      # MNIST images are 28x28 pixels, and have one color channel
      input_layer = tf.reshape(features["x"], [-1, self.config["image_size"], self.config["image_size"], 1])

      # Convolutional Layer #1
      # Computes 32 features using a 5x5 filter with ReLU activation.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 28, 28, 1]
      # Output Tensor Shape: [batch_size, 28, 28, 32]
      conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=self.config["conv_layer_0_filter"],
          kernel_size=[self.config["conv_layer_0_size"], self.config["conv_layer_0_size"]],
          padding="same",
          activation=tf.nn.relu)

      # Pooling Layer #1
      # First max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 28, 28, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 32]
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[self.config["pool_layer_0_size"], self.config["pool_layer_0_size"]], strides=self.config["pool_layer_0_stride"])

      # Convolutional Layer #2
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 14, 14, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 64]
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=self.config["conv_layer_1_filter"],
          kernel_size=[self.config["conv_layer_1_size"], self.config["conv_layer_1_size"]],
          padding="same",
          activation=tf.nn.relu)

      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 14, 14, 64]
      # Output Tensor Shape: [batch_size, 7, 7, 64]
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[self.config["pool_layer_1_size"], self.config["pool_layer_1_size"]], strides=self.config["pool_layer_1_stride"])

      # Flatten tensor into a batch of vectors
      # Input Tensor Shape: [batch_size, 7, 7, 64]
      # Output Tensor Shape: [batch_size, 7 * 7 * 64]
      flat_size = int(self.config["image_size"] / self.config["pool_layer_0_stride"] / self.config["pool_layer_1_stride"])
      pool2_flat = tf.reshape(pool2, [-1, flat_size * flat_size * self.config["conv_layer_1_filter"]])

      # Dense Layer
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 7 * 7 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

      # Add dropout operation; 0.6 probability that element will be kept
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

      # Logits layer
      # Input Tensor Shape: [batch_size, 1024]
      # Output Tensor Shape: [batch_size, 10]
      logits = tf.layers.dense(inputs=dropout, units=10)

      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

      # Calculate Loss (for both TRAIN and EVAL modes)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



    def train(self, epochs):


      # Create the Estimator
      mnist_classifier = tf.estimator.Estimator(
          model_fn=self.cnn_model_fn)

      # Set up logging for predictions
      # Log the values in the "Softmax" tensor with label "probabilities"
      tensors_to_log = {"probabilities": "softmax_tensor"}

      # Train the model
      train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": self.train_data},
          y=self.train_labels,
          batch_size=self.config["batch_size"],
          num_epochs=None,
          shuffle=True)
      mnist_classifier.train(
          input_fn=train_input_fn,
          steps=epochs)

      # Evaluate the model and print results
      eval_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": self.eval_data},
          y=self.eval_labels,
          num_epochs=1,
          shuffle=False)
      eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)

      return eval_results["accuracy"], eval_results["loss"]
