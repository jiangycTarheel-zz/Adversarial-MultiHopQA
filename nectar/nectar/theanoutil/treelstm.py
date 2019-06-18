"""Tree-LSTM encoder.

Based on Tai et al., 2015,
"Improved Semantic Representations From
Tree-Structured Long Short-Term Memory Networks."

Actually works on any DAG.

By convention, the root of the tree has only outgoing edges.
In DAG terminology, this means we start at sink nodes
and end at source nodes.
"""
import theano
from theano import tensor as T
from theano.ifelse import ifelse

import __init__ as ntu
from .. import log

def encode_child_sum(x_vecs, topo_order, adj_mat, c0, h0, W, U, Uf):
  """Run a child-sum tree-LSTM on a DAG.
  
  Args:
    x_vecs: n x e vector of node embeddings
    topo_order: a permutation of range(n) that gives a topological sort
        i.e. topo_order[i]'s children are in topo_order[:i]
    adj_mat: matrix where adj_mat[i,j] == 1 iff there is an i -> j edge.
    c0, h0, W, U, Uf: parameters of sizes 1 x d, 1 x d, e x 4d, d x 3d, d x d, respectively.
  """
  def recurrence(j, c_mat, h_mat, n, d, *args):
    x_j = x_vecs[j]
    children = T.eq(adj_mat[j,], 1).nonzero()  # let c(j) be number of children of node j
    c_children = c_mat[children]  # c(j) x d
    h_children = h_mat[children]  # c(j) x d
    # If this node has no children, use c0 and h0; else use c_mat and h_mat
    c_prev = ifelse(T.eq(c_children.shape[0], 0),
                    c0, c_children)  # max(c(j), 1) x d
    h_prev = ifelse(T.eq(h_children.shape[0], 0), 
                    h0, h_children)  # max(c(j), 1) x d

    h_tilde = T.sum(h_prev, axis=0)  # d
    w_prod = T.dot(x_j, W)  # 4d
    u_prod = T.dot(h_tilde, U)  # 3d
    uf_prod = T.dot(h_prev, Uf)  # max(c(j), 1) x d
    i_j = T.nnet.sigmoid(w_prod[:d] + u_prod[:d])  # d
    f_jk = T.nnet.sigmoid(w_prod[d:2*d] + uf_prod)  # c(j) x d
    o_j = T.nnet.sigmoid(w_prod[2*d:3*d] + u_prod[d:2*d])  # d
    u_j = T.tanh(w_prod[3*d:] + u_prod[2*d:])  # d
    fc_j = T.sum(f_jk * c_prev, axis=0)  # d
    c_j = i_j * u_j + fc_j
    h_j = o_j * T.tanh(c_j)

    # Update c_mat and h_mat
    new_c_mat = T.set_subtensor(c_mat[j], c_j)
    new_h_mat = T.set_subtensor(h_mat[j], h_j)
    return new_c_mat, new_h_mat

  n = x_vecs.shape[0]
  d = U.shape[0]
  (c_list, h_list), _ = theano.scan(
      recurrence, sequences=[topo_order],
      outputs_info=[T.zeros((n, d)), T.zeros((n, d))],
      non_sequences=[n, d, x_vecs, adj_mat, c0, h0, W, U, Uf])
  return c_list[-1], h_list[-1]
