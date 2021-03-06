# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""

    self.vocab_size = 8800

    # Batch size.
    self.batch_size = 64

    # Scale used to initialize model variables.
    self.initializer_scale = 0.01#0.08

    # LSTM input and output dimensionality, respectively.
    self.embedding_size = 32#50#512#256#128#512
    self.num_lstm_units = 512#50#512#256#128#512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 1#0.5 #0.7


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 566747# old :586363

    # Optimizer for training the model.
    self.optimizer = "Adam" #"SGD"

    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 0.0002#2.0
    self.learning_rate_decay_factor = 0.25
    self.num_epochs_per_decay = 8.0

    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0
    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5

