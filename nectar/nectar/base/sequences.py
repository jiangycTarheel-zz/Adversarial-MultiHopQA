"""Utilities related to sequences."""
def edit_distance(x1, x2, dist_func=None, gap_penalty=1):
  """Compute edit distance between two sequences.
  
  Args:
    x1: First sequence.
    x2: Second sequence.
    dist_func: Distance score between two tokens (default = Levenstein).
    gap_penalty: Penalty for gaps (default = 1, for Levenstein)
  """
  if not dist_func:
    dist_func = lambda x, y: int(x != y)
  n1, n2 = len(x1), len(x2)
  scores = [[i * gap_penalty for i in range(n2+1)]]
  ptrs = [[None] + [(0, i-1) for i in range(1, n2+1)]]
  for i in range(1, n1 + 1):
    cur_scores = [scores[i-1][0] + gap_penalty]
    cur_ptrs = [(i-1, 0)]
    for j in range(1, n2 + 1):
      local_score = dist_func(x1[i-1], x2[j-1])
      poss_scores = [scores[i-1][j-1] + local_score,
                    scores[i-1][j] + gap_penalty,
                    cur_scores[j-1] + gap_penalty]
      score_ind, cur_score = min(enumerate(poss_scores), key=lambda x: x[1])
      cur_scores.append(cur_score)
      if score_ind == 0:
        cur_ptr = (i-1, j-1)
      elif score_ind == 1:
        cur_ptr = (i-1, j)
      else:
        cur_ptr = (i, j-1)
      cur_ptrs.append(cur_ptr)
    scores.append(cur_scores)
    ptrs.append(cur_ptrs)
  dist = scores[n1][n2]
  return dist, ptrs

def get_unaligned_spans(x1, x2, ptrs):
  """Get unaligned spans from an edit distance pointer matrix."""
  n1, n2 = len(x1), len(x2)
  i1, i2 = n1, n2
  spans = []
  end1, end2 = None, None
  while i1 != 0 or i2 != 0:
    i1_new, i2_new = ptrs[i1][i2]
    if i1_new == i1 or i2_new == i2 or x1[i1_new] != x2[i2_new]:  # mismatch/gap
      if end1 is None:
        end1, end2 = i1, i2
    else:
      if end1 is not None:
        spans.append(((i1, end1), (i2, end2)))
        end1, end2 = None, None
    i1, i2 = i1_new, i2_new
  if end1 is not None:
    spans.append(((0, end1), (0, end2)))
  return spans
