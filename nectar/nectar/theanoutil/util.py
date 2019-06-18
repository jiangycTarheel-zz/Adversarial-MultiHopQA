"""Some common theano utilities."""
import numpy as np
import theano
from theano import tensor as T
from theano.ifelse import ifelse

def printed(var, name=''):
  return theano.printing.Print(name)(var)

def logsumexp(mat, axis=0):
  """Apply a row-wise log-sum-exp, summing along axis."""
  maxes = T.max(mat, axis=axis)
  return T.log(T.sum(T.exp(mat - maxes), axis=axis)) + maxes

def clip_gradients(grads, clip_thresh):
  """Clip gradients to some total norm."""
  total_norm = T.sqrt(sum(T.sum(g**2) for g in grads))
  scale = ifelse(T.gt(total_norm, clip_thresh),
                 clip_thresh / total_norm, 
                 np.dtype(theano.config.floatX).type(1.0))
  clipped_grads = [scale * g for g in grads]
  return clipped_grads

def create_grad_cache(param_list, name='grad_cache'):
  """Create a grad cache, for things like momentum or AdaGrad."""
  cache = [theano.shared(name='%s_%s' % (p.name, name),
                         value=np.zeros_like(p.get_value()))
           for p in param_list]
  return cache

def get_vanilla_sgd_updates(param_list, gradients, lr):
  """Do SGD updates with vanilla step rule.""" 
  updates = []
  for p, g in zip(param_list, gradients):
    new_p = p - lr * g
    has_non_finite = T.any(T.isnan(new_p) + T.isinf(new_p))
    updates.append((p, ifelse(has_non_finite, p, new_p)))
  return updates

def get_nesterov_sgd_updates(param_list, gradients, velocities, lr, mu):
  """Do SGD updates with Nesterov momentum.""" 
  updates = []
  for p, g, v in zip(param_list, gradients, velocities):
    new_v = mu * v - lr * g
    new_p = p - mu * v + (1 + mu) * new_v
    has_non_finite = (T.any(T.isnan(new_p) + T.isinf(new_p)) +
                      T.any(T.isnan(new_v) + T.isinf(new_v)))
    updates.append((p, ifelse(has_non_finite, p, new_p)))
    updates.append((v, ifelse(has_non_finite, v, new_v)))
  return updates

def plot_learning_curve(data, outfile=None):
  if outfile:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
  else:
    import matplotlib.pyplot as plt 

  plt.figure(figsize=(12, 5)) 
  for i, (name, cur_data) in enumerate(data):
    plt.subplot(1, len(data), i+1)
    plt.plot(cur_data)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.ylim(0, 1.1 * max(cur_data))

  if outfile:
    plt.savefig(outfile)
  else:
    plt.show()
