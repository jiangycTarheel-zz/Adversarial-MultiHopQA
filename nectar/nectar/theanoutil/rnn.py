"""Common RNN-related functions."""
import theano
from theano import tensor as T
from util import printed

def lstm_split(c_h):
  """Split the joint c_t and h_t of the LSTM state."""
  d = c_h.shape[0]/2
  c = c_h[:d]
  h = c_h[d:]
  return c, h

def lstm_step(c_h_prev, input_t, W_mat):
  """The LSTM recurrence.
  
  Args:
    c_h_prev: A vector of size 2d, concatenation of
        memory cell c_prev and hidden state h_prev
    input_t: Current input, as a vector of size e
    W_mat: transition matrix of size (d+e) x (4d)
  """
  c_prev, h_prev = lstm_split(c_h_prev)
  d = c_prev.shape[0]
  vec_t = T.concatenate([h_prev, input_t])
  prod = T.dot(vec_t, W_mat)
  i_t = T.nnet.sigmoid(prod[:d])
  f_t = T.nnet.sigmoid(prod[d:2*d])
  o_t = T.nnet.sigmoid(prod[2*d:3*d])
  c_tilde_t = T.tanh(prod[3*d:])
  c_t = f_t * c_prev + i_t * c_tilde_t
  h_t = o_t * T.tanh(c_t)
  c_h_t = T.concatenate([c_t, h_t])
  return c_h_t

def batch_lstm_split(c_h):
  """Split the joint c_t and h_t of the LSTM state (batch mode)."""
  d = c_h.shape[1]/2
  c = c_h[:,:d]
  h = c_h[:,d:]
  return c, h

def time_batch_lstm_split(c_h):
  """Split the joint c_t and h_t of the LSTM state (time + batch mode)."""
  d = c_h.shape[2]/2
  c = c_h[:,:,:d]
  h = c_h[:,:,d:]
  return c, h

def batch_lstm_step(c_h_prev, input_t, W_mat):
  """The LSTM recurrence (batch mode).
  
  Args:
    c_h_prev: matrix of size batch_sz x 2d, concatenation of
        memory cell c_prev and hidden state h_prev
    input_t: Current input, as matrix of size batch_sz x e
    W_mat: transition matrix of size (d+e) x (4d)
  """
  c_prev, h_prev = batch_lstm_split(c_h_prev)  # batch_sz x d each
  d = c_prev.shape[1]
  vec_t = T.concatenate([h_prev, input_t], axis=1)  # batch_sz x (d+e)
  prod = T.dot(vec_t, W_mat)  # batch_sz x 4d
  i_t = T.nnet.sigmoid(prod[:,:d])
  f_t = T.nnet.sigmoid(prod[:,d:2*d])
  o_t = T.nnet.sigmoid(prod[:,2*d:3*d])
  c_tilde_t = T.tanh(prod[:,3*d:])
  c_t = f_t * c_prev + i_t * c_tilde_t
  h_t = o_t * T.tanh(c_t)
  c_h_t = T.concatenate([c_t, h_t], axis=1)  # batch_sz x 2d
  return c_h_t
