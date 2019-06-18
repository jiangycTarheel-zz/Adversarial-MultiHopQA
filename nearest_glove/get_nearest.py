"""
Adapted from https://github.com/brannondorsey/GloVe-experiments
"""

import argparse, sys, readline
from scipy.spatial.distance import cosine
from nearest_glove.utils import build_word_vector_matrix, get_label_dictionaries
from difflib import SequenceMatcher
from nltk.corpus import wordnet

def word_arithmetic(start_word, minus_words, plus_words, word_to_id, id_to_word, df, num_results=5):
  '''Returns a word string that is the result of the vector arithmetic'''
  try:
    start_vec  = df[word_to_id[start_word]]
    minus_vecs = [df[word_to_id[minus_word]] for minus_word in minus_words]
    plus_vecs  = [df[word_to_id[plus_word]] for plus_word in plus_words]
  except KeyError as err:
    return err, None

  result = start_vec

  if minus_vecs:
    for i, vec in enumerate(minus_vecs):
      result = result - vec

  if plus_vecs:
    for i, vec in enumerate(plus_vecs):
      result = result + vec

  # result = start_vec - minus_vec + plus_vec
  words = [start_word] + minus_words + plus_words
  return None, find_nearest(words, result, id_to_word, df, num_results)

def find_nearest(words, vec, id_to_word, df, num_results, method='cosine'):

  if method == 'cosine':
    minim = [] # min, index
    for i, v in enumerate(df):
      # skip the base word, its usually the closest
      if id_to_word[i] in words:
        continue
      dist = cosine(vec, v)
      minim.append((dist, i))
    minim = sorted(minim, key=lambda v: v[0])
    # return list of (word, cosine distance) tuples
    return [(id_to_word[minim[i][1]], minim[i][0]) for i in range(num_results)]
  else:
    raise Exception('{} is not an excepted method parameter'.format(method))

def longest_common_substring(string1, string2):
  match_obj = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
  return match_obj.size

def find_nearest_word(start_word, df, word_to_id, id_to_word, num_results=5):
  err, results = word_arithmetic (start_word=start_word,
                                  minus_words=[],
                                  plus_words=[],
                                  word_to_id=word_to_id,
                                  id_to_word=id_to_word,
                                  df=df,
                                  num_results=num_results)
  if results: 
    res = [t[0] for t in results if t[0] not in start_word and start_word not in t[0] and longest_common_substring(t[0], start_word) < 4]
    if len(res) > 0:
      return res
    else:
      return None
  else:
    return None


def build_glove_matrix(num_words=40000):
  vector_file = 'data/glove/glove.6B.100d.txt'
  df, labels_array = build_word_vector_matrix(vector_file, num_words)
  word_to_id, id_to_word = get_label_dictionaries(labels_array)
  return df, word_to_id, id_to_word 
