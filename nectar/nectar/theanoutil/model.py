"""Standard utilties for a theano model."""
import collections
import numbers
import numpy as np
import pickle
import random
import sys
import theano
import time
from Tkinter import TclError

import __init__ as ntu
from .. import log, secs_to_str

class TheanoModel(object):
  """A generic theano model.

  This class handles some standard boilerplate.
  Current features include:
    Basic training loop
    Saving and reloading of model

  An implementing subclass must override the following methods:
    self.__init__(*args, **kwargs)
    self.init_params()
    self.setup_theano_funcs()
    self.get_metrics(example)
    self.train_one(example, lr)
    self.evaluate(dataset)

  A basic self.__init__() routine is provided here, just as an example.
  Most users should override __init__() to perform additional functionality.

  See these methods for more details.
  """
  def __init__(self):
    """A minimal example of what functions must be called during initialization.

    Implementing subclasses should override this method,
    but maintain the basic functionality presented here.
    """
    # self.params, self.params_list, and self.param_names are required by self.create_matrix()
    self.params = {}
    self.param_list = []
    self.param_names = []

    # Initialize parameters
    self.init_params()

    # If desired, set up grad norm caches for momentum, AdaGrad, etc. here.
    # It must be after params are initialized but before theano functionss are created.
    # self.velocities = nt.create_grad_cache(self.param_list)

    # Set up theano functions
    self.theano_funcs = {} 
    self.setup_theano_funcs() 

  def init_params(self):
    """Initialize parameters with repeated calls to self.create_matrix()."""
    raise NotImplementedError

  def setup_theano_funcs(self):
    """Create theano.function objects for this model in self.theano_funcs."""
    raise NotImplementedError

  def get_metrics(self, example):
    """Get accuracy metrics on a single example.

    Args:
      example: An example (possibly batch)
      lr: Current learning rate
    Returns:
      dictionary mapping metric_name to (value, weight);
      |weight| is used to compute weighted average over dataset.
    """
    raise NotImplementedError

  def train_one(self, example, lr):
    """Run training on a single example.
    
    Args:
      example: An example (possibly batch)
      lr: Current learning rate
    Returns:
      dictionary mapping metric_name to (value, weight);
      |weight| is used to compute weighted average over dataset.
    """
    raise NotImplementedError

  def create_matrix(self, name, shape, weight_scale, value=None):
    """A helper method that creates a parameter matrix."""
    if value:
      pass
    elif shape:
      value = weight_scale * np.random.uniform(-1.0, 1.0, shape).astype(
          theano.config.floatX)
    else:
      # None means it's a scalar
      dtype = np.dtype(theano.config.floatX)
      value = dtype.type(weight_scale * np.random.uniform(-1.0, 1.0))
    mat = theano.shared(name=name, value=value)
    self.params[name] = mat
    self.param_list.append(mat)
    self.param_names.append(name)

  def train(self, train_data, lr_init, epochs, dev_data=None, rng_seed=0,
            plot_metric=None, plot_outfile=None):
    """Train the model.

    Args:
      train_data: A list of training examples
      lr_init: Initial learning rate
      epochs: An integer number of epochs to train, or a list of integers, 
          where we halve the learning rate after each period.
      dev_data: A list of dev examples, evaluate loss on this each epoch.
      rng_seed: Random seed for shuffling the dataset at each epoch.
      plot_metric: If True, plot a learning for the given metric.
      plot_outfile: If provided, save learning curve to file.
    """
    random.seed(rng_seed)
    train_data = list(train_data)
    lr = lr_init
    if isinstance(epochs, numbers.Number):
      lr_changes = []
      num_epochs = epochs
    else:
      lr_changes = set([sum(epochs[:i]) for i in range(1, len(epochs))])
      num_epochs = sum(epochs)
    num_epochs_digits = len(str(num_epochs))
    train_plot_list = []
    dev_plot_list = []
    str_len_dict = collections.defaultdict(int)
    len_time = 0
    for epoch in range(num_epochs):
      t0 = time.time()
      random.shuffle(train_data)
      if epoch in lr_changes:
        lr *= 0.5
      train_metric_list = []
      for ex in train_data:
        cur_metrics = self.train_one(ex, lr)
        train_metric_list.append(cur_metrics)
      if dev_data:
        dev_metric_list = [self.get_metrics(ex) for ex in dev_data]
      else:
        dev_metric_list = []
      t1 = time.time()

      # Compute the averaged metrics
      train_metrics = aggregate_metrics(train_metric_list)
      dev_metrics = aggregate_metrics(dev_metric_list)
      if plot_metric:
        train_plot_list.append(train_metrics[plot_metric])
        if dev_metrics:
          dev_plot_list.append(dev_metrics[plot_metric])

      # Some formatting to make things align in columns
      train_str = format_epoch_str('train', train_metrics, str_len_dict)
      dev_str = format_epoch_str('dev', dev_metrics, str_len_dict)
      metric_str = ', '.join(x for x in [train_str, dev_str] if x)
      time_str = secs_to_str(t1 - t0)
      len_time = max(len(time_str), len_time)
      log('Epoch %s: %s [lr = %.1e] [took %s]' % (
          str(epoch+1).rjust(num_epochs_digits), metric_str, lr,
          time_str.rjust(len_time)))

    if plot_metric:
      plot_data = [('%s on train data' % plot_metric, train_plot_list)]
      if dev_plot_list:
        plot_data.append(('%s on dev data' % plot_metric, dev_plot_list))
      try:
        ntu.plot_learning_curve(plot_data, outfile=plot_outfile)
      except TclError: 
        print >> sys.stderr, 'Encoutered error while plotting learning curve'


  def evaluate(self, dataset):
    """Evaluate the model."""
    metrics_list = [self.get_metrics(ex) for ex in dataset]
    return aggregate_metrics(metrics_list)

  def save(self, filename):
    # Save
    tf = self.theano_funcs
    params = self.params
    param_list = self.param_list
    # Don't pickle theano functions
    self.theano_funcs = None
    # CPU/GPU portability
    self.params = {k: v.get_value() for k, v in params.iteritems()}
    self.param_list = None
    # Any other things to do before saving
    saved = self._prepare_save()
    with open(filename, 'wb') as f:
      pickle.dump(self, f)
    # Restore
    self.theano_funcs = tf
    self.params = params
    self.param_list = param_list
    self._after_save(saved)

  def _prepare_save(self):
    """Any additional things before calling pickle.dump()."""
    pass

  def _after_save(self, saved):
    """Any additional things after calling pickle.dump()."""
    pass

  @classmethod
  def load(cls, filename):
    with open(filename, 'rb') as f:
      model = pickle.load(f)
    # Recreate theano shared variables
    params = model.params
    model.params = {}
    model.param_list = []
    for name in model.param_names:
      value = params[name]
      mat = theano.shared(name=name, value=value)
      model.params[name] = mat
      model.param_list.append(mat)
    model._after_load()
    # Recompile theano functions
    model.theano_funcs = {}
    model.setup_theano_funcs()
    return model

  def _after_load(self):
    """Any additional things after calling pickle.load()."""
    pass

def aggregate_metrics(metric_list):
  metrics = collections.OrderedDict()
  if metric_list: 
    keys = metric_list[0].keys()
    for k in keys:
      numer = sum(x[k][0] * x[k][1] for x in metric_list)
      denom = sum(x[k][1] for x in metric_list)
      metrics[k] = float(numer) / denom
  return metrics

def format_epoch_str(name, metrics, str_len_dict):
  if not metrics: return ''
  toks = []
  for k in metrics:
    val_str = '%.4f' % metrics[k]
    len_key = '%s:%s' % (name, k)
    str_len_dict[len_key] = max(str_len_dict[len_key], len(val_str))
    cur_str = '%s=%s' % (k, val_str.rjust(str_len_dict[len_key]))
    toks.append(cur_str)
  return '%s(%s)' % (name, ', '.join(toks))

