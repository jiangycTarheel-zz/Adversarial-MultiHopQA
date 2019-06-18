"""A basic trie."""
import argparse
import sys

class Trie(object):
  def __init__(self):
    self.root = {}

  def add(self, seq):
    node = self.root
    for i, x in enumerate(seq):
      if x not in node:
        node[x] = (False, {})
      if i == len(seq) - 1:
        node[x] = (True, node[x][1])
      else:
        is_terminal, node = node[x]

  def remove(self, seq):
    node = self.root
    nodes = []
    for i, x in enumerate(seq):
      nodes.append(node)
      if x not in node:
        raise ValueError('Item not found, cannot be removed')
      if i == len(seq) - 1:
        # Actually remove
        node[x] = (False, node[x][1])
      else:
        is_terminal, node = node[x]
    # Clean up
    for i in range(len(seq) - 1, -1, -1):
      # nodes[i] contains seq[i]
      node = nodes[i]
      x = seq[i]
      is_terminal, next_node = node[x]
      if not is_terminal and not next_node:
        del node[x]
      else:
        break

  def contains(self, seq):
    node = self.root
    for x in seq:
      if x not in node:
        return False
      is_terminal, node = node[x]
    return is_terminal

  def contains_prefix(self, seq):
    node = self.root
    for x in seq:
      if x not in node:
        return False
      is_terminal, node = node[x]
    return True
    
  def get_node(self, seq):
    node = self.root
    for x in seq:
      if x not in node:
        return None
      is_terminal, node = node[x]
    return node

  def __iter__(self):
    stack = [((), self.root)]
    while stack:
      prefix, node = stack.pop()
      for k in node:
        new_prefix = prefix + (k,)
        is_terminal, new_node = node[k]
        if is_terminal:
          yield new_prefix
        stack.append((new_prefix, new_node))

def main():
  trie = Trie()
  print 'Running basic tests...'
  trie.add((0,))
  trie.add((1, 2, 3))
  assert trie.contains((0,)) == True
  assert trie.contains((1, 2, 3)) == True
  assert trie.contains((1,)) == False
  assert trie.contains_prefix((1,)) == True
  assert trie.contains((1, 2)) == False
  assert trie.contains_prefix((1, 2)) == True
  assert trie.contains((2,)) == False
  trie.add((1, 2))
  trie.add((1, 4))
  trie.add((5, 6))
  assert trie.contains((1, 2, 3)) == True
  assert trie.contains((1, 2)) == True
  assert trie.contains_prefix((1, 2)) == True
  assert trie.contains((2,)) == False
  assert trie.contains_prefix((2,)) == False
  assert trie.contains((5,)) == False
  assert trie.contains((1, 4)) == True
  assert trie.contains((5, 6)) == True
  assert trie.contains_prefix((5,)) == True
  trie.remove((1, 2, 3))
  assert trie.contains((1, 2, 3)) == False
  assert trie.contains((1, 2)) == True
  assert trie.contains_prefix((1, 2)) == True
  trie.add((1, 2, 3))
  trie.remove((1, 2))
  trie.add((1,))
  assert trie.contains((1, 2, 3)) == True
  assert trie.contains((1, 2)) == False
  assert trie.contains((1,)) == True
  assert trie.contains_prefix((1, 2)) == True
  assert set(trie) == set([(0,), (1,), (1, 2, 3), (1, 4), (5, 6)])
  print trie.root
  print 'All pass!'

if __name__ == '__main__':
  main()
