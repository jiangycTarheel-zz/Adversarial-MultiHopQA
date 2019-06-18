"""Utilities for handling fig LispTree objects."""

def tokenize(s):
  toks = []
  cur_chars = []
  inside_str = False
  inside_escape = False
  for c in s:
    if inside_str:
      if inside_escape:
        inside_escape = False
        cur_chars.append(c)
      else:
        if c == '"':
          inside_str = False
          toks.append(''.join(cur_chars))
          cur_chars = []
        elif c == '\\':
          inside_escape = True
        else:
          cur_chars.append(c)
    else:
      if inside_escape:
        inside_escape = False
        cur_chars.append(c)
      else:
        if c in ('(', ')'):
          if cur_chars:
            toks.append(''.join(cur_chars))
            cur_chars = []
          toks.append(c)
        elif c == ' ':
          if cur_chars:
            toks.append(''.join(cur_chars))
            cur_chars = []
        elif c == '"':
          if cur_chars:
            raise ValueError('" character found in middle of token')
          inside_str = True
        elif c == '\\':
          inside_escape = True
        else:
          cur_chars.append(c)
  if cur_chars:
    toks.append(''.join(cur_chars))
  return toks

def from_string(s):
  """Parse a Java fig LispTree from a string."""
  toks = tokenize(s)
  def recurse(i):
    if toks[i] == '(':
      subtrees = []
      j = i+1 
      while True:
        subtree, j = recurse(j)
        subtrees.append(subtree)
        if toks[j] == ')':
          return tuple(subtrees), j + 1 
    else:
      return toks[i], i+1 
  lisp_tree, final_ind = recurse(0)
  return lisp_tree
