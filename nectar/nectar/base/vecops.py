"""Operations on sparse vectors represented as dicts."""
import collections

# Mutating a vector
def add(v, other, scale=1):
  for k in other:
    if k in v:
      v[k] += scale * other[k]
    else:
      v[k] = scale * other[k]

def scale(v, scale):
  for k in v.keys():
    v[k] *= scale

# Returning a new vector or scalar
def dot(v1, v2):
  if len(v1) > len(v2):
    return dot(v2, v1)
  ans = 0
  for k in v1:
    if k in v2:
      ans += v1[k] * v2[k]
  return ans

def sum(v1, v2):
  ans = collections.defaultdict(float)
  for k in v1:
    ans[k] += v1[k]
  for k in v2:
    ans[k] += v2[k]
  return ans

def l2norm(v):
  return math.sqrt(sum(v[k]**2 for k in v))
